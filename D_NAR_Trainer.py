#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 11:14:23 2025

@author: root
"""

import torch
import librosa
from torch.utils.data import  DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import  destroy_process_group
import dac 
from Models.D_NAR import D_NAR_Model
from Trainer import Trainer, labled_AudioDataset, cosine_decay, ddp_setup, par_count , compute_sisdr
import tqdm
from einops import rearrange
import numpy as np
from pystoi import stoi
import soundfile as sf



"""

TRAINING PARAMETERS

"""
device='cuda'
DAC_Model ="DAC_MODELS/weights_16khz.pth"
DATA_PATHS = [
    "Path/to/train-360/s1",
    "Path/to/train-360/mix_single",
    "Path/to/dev/s1",
    "Path/to/dev/mix_single",
    ]

sr=16000
NUM_EPOCHS =300
BATCH_SIZE =32
SAVE_EVERY=50
GEN_EVERY=25
LEARNING_RATE =0.005 * (BATCH_SIZE/256)
checkpoint_path=None #"Checkpoints/Model_Chechpoint.pt"

"""

MODEL HYPER-PARAMETERS

"""
NAME="Model_Name"

MAX_LEN =50
Nq=12
Params = {"num_tokens":1024,
       "dim":384,
       "max_seq_len":MAX_LEN,
       "N_layers":8,
       "dim_head":32,
       "heads":12 }
    

def load_train_objs( DAC_Model, device,Params,LEARNING_RATE,checkpoint_path,NUM_EPOCHS):

    DAC_Model = dac.DAC.load(DAC_Model)
    DAC_Model.encoder.to(device)
    DAC_Model.quantizer.to(device)
    
    if checkpoint_path:
        print("checkpoint found : " , checkpoint_path)
        checkpoint = torch.load(checkpoint_path,map_location=device,weights_only=False)
        SE_Model_params = checkpoint['SE_Model_params']
    else : 
        SE_Model_params = Params
    SE_Model= D_NAR_Model(**SE_Model_params).to(device)
    optimizer = torch.optim.AdamW(SE_Model.parameters(),
                                  lr= LEARNING_RATE,
                                  betas=(0.9, 0.95),
                                  weight_decay=0.05)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                     lr_lambda=lambda epoch: cosine_decay(epoch,10, NUM_EPOCHS))

    return DAC_Model, SE_Model, optimizer, lr_scheduler



class D_NAR_Trainer(Trainer):
    def _denoise_validation(self,epoch):
            """
            Performs validation specifically for the denoising phase.
    
            Args:
                epoch: Current epoch number.
    
            """
            self.SE_Model.module.eval()
            with torch.no_grad():
                SI_SDR = []
                ESTOI =[]
                k=np.random.randint(self.val_dataset.__len__())
                for i , data in enumerate(tqdm.tqdm(self.val_dataset, desc="Validation : Denoising")):
                    
                    x_n = self.Tokenize(data[1].to(self.gpu_id))
                    
                    z_gen=self.SE_Model.module.generate(noisy=x_n)
                    z_gen = rearrange(z_gen,"b t n -> b n t")
                    y=self.Decode(self.Detokenize(z_gen))
                    y = y.detach().cpu().numpy().squeeze()
                    xc = data[0].detach().cpu().numpy().squeeze()
                    x_n = data[1].detach().cpu().numpy().squeeze()
                    for j, x_i in enumerate(xc):
                        try:
                            x_i=x_i[:y[j].shape[0]]
                            SI_SDR.append(compute_sisdr(y[j], x_i))
                            ESTOI.append(stoi(x_i, y[j], 16000, extended=True))
                        except Exception as e:
                            print(f"STOI computation failed for batch {i} : {j}: {e}")
                            break
                        if (j ==1 ) and (i==k) :
                           sf.write(f"{self.NAME}_Audio/Reconstructed_Denoised_{epoch}.wav", y[j] , 16000)
                           sf.write(f"{self.NAME}_Audio/Noisy_Signal{epoch}.wav",            x_n[j], 16000)
                           sf.write(f"{self.NAME}_Audio/Clean_Signal_{epoch}.wav", x_i, 16000)

            print(f"  SI_SDR computation Complete : SI_SDR = {np.mean(SI_SDR)} for Epoch = {epoch}")
            print(f"  ESTOI  computation Complete : ESTOI  = {np.mean(ESTOI)} for Epoch = {epoch}")
            
            
    def _save_checkpoint(self, epoch):
        SE_Model_params = {
            'num_tokens':self.SE_Model.module.num_tokens,
            'dim': self.SE_Model.module.dim,
            'max_seq_len': self.SE_Model.module.max_seq_len,
            'N_layers': len(self.SE_Model.module.noise_transformer.layers),
            'dim_head': int(self.SE_Model.module.dim/self.SE_Model.module.noise_transformer.layers[0].heads),
            'heads':self.SE_Model.module.noise_transformer.layers[0].heads
        }
        ckp = self.SE_Model.module.state_dict()
        opt = self.optimizer.state_dict()
        PATH = f"Checkpoints/{self.NAME}_ckpt_{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict':ckp,
            'optimizer_state_dict': opt,
            'SE_Model_params': SE_Model_params
            }, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")



def main(rank: int,
         world_size: int,
         DAC_Model,
         device,
         DATA_PATHS,
         SAVE_EVERY: int,
         GEN_EVERY:int,
         NUM_EPOCHS: int,
         BATCH_SIZE: int,
         LEARNING_RATE,
         Nq,
         MAX_LEN ,
         Params,
         checkpoint_path,
         NAME,
         sr
         ):
    
    ddp_setup(rank, world_size)
    max_len=int(sr*(MAX_LEN/50))
    clean_training= librosa.util.find_files( DATA_PATHS[0], ext='wav')[:10]
    noisy_training= librosa.util.find_files(  DATA_PATHS[1], ext='wav')[:10] 
    dataset=labled_AudioDataset(clean_training, noisy_training, max_len,random_start=False)
    train_dataset=DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, sampler=DistributedSampler(dataset))
    clean_validation= librosa.util.find_files( DATA_PATHS[2], ext='wav')[:10] 
    noisy_validation= librosa.util.find_files( DATA_PATHS[3], ext='wav')[:10] 
    dataset=labled_AudioDataset(clean_validation, noisy_validation, max_len,random_start=False)
    val_dataset=DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,sampler=DistributedSampler(dataset))
    
    DAC_Model, SE_Model, optimizer, scheduler = load_train_objs(DAC_Model,device,Params,LEARNING_RATE,checkpoint_path,NUM_EPOCHS)
    
    
    parameter_count=par_count(SE_Model)
    print("Number of Model Parameters :",parameter_count)

    trainer = D_NAR_Trainer(  DAC_Model
                      , SE_Model
                      , rank
                      , train_dataset
                      , val_dataset
                      , optimizer
                      , scheduler
                      , SAVE_EVERY
                      , GEN_EVERY
                      , Nq=Nq
                      , sr=sr
                      , NAME=NAME)
    
    if checkpoint_path:
       start_epoch = trainer.load_from_checkpoint(checkpoint_path)
    else:
       start_epoch = 0
    trainer.train(NUM_EPOCHS, start_epoch )
    destroy_process_group()

if __name__ == '__main__':

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size,
                         DAC_Model,
                         device,
                         DATA_PATHS,
                         SAVE_EVERY,
                         GEN_EVERY,
                         NUM_EPOCHS,
                         BATCH_SIZE,
                         LEARNING_RATE,
                         Nq,
                         MAX_LEN,
                         Params,
                         checkpoint_path,
                         NAME,
                         sr ), nprocs=world_size)
