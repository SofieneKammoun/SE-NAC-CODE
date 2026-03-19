#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 11:14:23 2025

@author: root
"""




import os
import torch
import librosa
from torch.utils.data import  DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import  destroy_process_group
from Models.STFT_NAR_Mask import  C_NAR_Model
from Trainer import Trainer, labled_AudioDataset, cosine_decay, ddp_setup, par_count , compute_sisdr
import tqdm
from einops import rearrange
import numpy as np
from pystoi import stoi
import soundfile as sf

from torch.nn.parallel import DistributedDataParallel as DDP



"""

TRAINING PARAMETERS

"""
device='cuda'
DATA_PATHS = [
    "Path/to/train-360/s1",
    "Path/to/train-360/mix_single",
    "Path/to/dev/s1",
    "Path/to/dev/mix_single",
    ] 

sr=16000
NUM_EPOCHS =300
BATCH_SIZE =256
SAVE_EVERY=50
GEN_EVERY=20
LEARNING_RATE =0.0001 * (BATCH_SIZE/256)
checkpoint_path=None #"Checkpoints/Model_Chechpoint.pt"

NAME="STFT_Model_Mask"

"""

MODEL HYPER-PARAMETERS

"""

MAX_LEN =80
Params = {"input_dim":1026,
       "dim":384,
       "max_seq_len":MAX_LEN,
       "N_layers":16,
       "dim_head":32,
       "heads":12 }
n_fft=1024
hop_length=256
win_length=n_fft
    


def load_train_objs( device,Params,LEARNING_RATE,checkpoint_path,NUM_EPOCHS):

    if checkpoint_path:
        print("checkpoint found : " , checkpoint_path)
        checkpoint = torch.load(checkpoint_path,map_location=device,weights_only=False)
        SE_Model_params = checkpoint['SE_Model_params']
    else : 
        SE_Model_params = Params
    SE_Model= C_NAR_Model(**SE_Model_params).to(device)
    optimizer = torch.optim.AdamW(SE_Model.parameters(),
                                  lr= LEARNING_RATE,
                                  betas=(0.9, 0.95),
                                  weight_decay=0.05)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                     lr_lambda=lambda epoch: cosine_decay(epoch,10, NUM_EPOCHS))

    return  SE_Model, optimizer, lr_scheduler



class STFT_NAR_Trainer:
    def __init__(self, SE_Model, gpu_id, dataset,Val_dataset,optimizer, scheduler, save_every,gen_every,sr,NAME):
        
        """
        Trainer class.

        Args:
            SE_Model: The RQ-Transformer model.
            gpu_id: The GPU device ID to use (rank).
            dataset:     Training dataset.
            Val_dataset: Validation dataset.
            optimizer: Optimizer for model training.
            scheduler: Learning rate scheduler.
            save_every: Interval (in epochs) to save model checkpoints.
            gen_every: Interval (in epochs) to generate samples.
            sr: Sample rate for audio processing.

        Initializes distributed data parallel (DDP) models
        """
        self.SE_Model = SE_Model.to(gpu_id)
        self.gpu_id = gpu_id
        self.dataset = dataset
        self.val_dataset = Val_dataset
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_every = save_every
        self.gen_every = gen_every
        self.sr=sr
        self.total_steps = 0
        self.loss_values = []
        self.loss_avg = []
        self.val = []
        self.lr = []
        self.SE_Model = DDP(SE_Model, device_ids=[gpu_id],find_unused_parameters=True)
        self.NAME=NAME

    def Encode(self,x):
         
         
        window = torch.hann_window(win_length).to(self.gpu_id)
        stft = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            return_complex=True
        )
        return torch.cat((stft.real,stft.imag),dim=1)





    def Decode(self, combined):
        real, imag= torch.split(combined, combined.shape[1] // 2, dim=1)
        stft=torch.complex(real,imag).to(self.gpu_id)
        window = torch.hann_window(win_length).to(self.gpu_id)

        reconstructed = torch.istft(
            stft,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
        )
        return reconstructed



    def process_batch_train_audio(self, batch):
        
        x=self.Encode(batch[0].squeeze(1).to(self.gpu_id))
        y=self.Encode(batch[1].squeeze(1).to(self.gpu_id))
        y=rearrange(y, "b d t -> b t d")
        x=rearrange(x, "b d t -> b t d")
        loss = self.SE_Model(embeds=y,clean_embeds=x,return_loss=True)
        return loss


    def _run_batch(self, batch,epoch):
        self.optimizer.zero_grad()
        loss = self.process_batch_train_audio(batch)
        loss.backward()
        self.optimizer.step()
        return loss.item()


    def _denoise_validation(self,epoch):
        """
        Performs validation specifically for the denoising phase.

        Args:
            epoch: Current epoch number.

        Computes SI-SDR and ESTOI metrics for evaluating speech enhancement.
        """
        self.SE_Model.module.eval()
        with torch.no_grad():
            SI_SDR = []
            ESTOI =[]
            k=np.random.randint(self.val_dataset.__len__())

            for i , data in enumerate(tqdm.tqdm(self.val_dataset, desc="Validation : Denoising")):
                x_n=data[1].squeeze(1).to(self.gpu_id)
                y=self.Encode(x_n)
                x=self.Encode(data[0].squeeze(1).to(self.gpu_id))

                y=rearrange(y, "b d t -> b t d")
                x=rearrange(x, "b d t -> b t d")
                
                y_ = self.SE_Model(embeds=y,clean_embeds=x,return_loss=False)
                y_=rearrange(y_ ,'b t d-> b d t')
                y = self.Decode(y_)
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
                    
 

    def _validate(self ):
        self.SE_Model.eval()
        with torch.no_grad():
            loss_values = []
            for i , data in enumerate(tqdm.tqdm(self.val_dataset, desc="Validation :")):
                    loss = self.process_batch_train_audio(data)
                    loss_values.append(loss.item())
        return np.mean(loss_values)


    def _run_epoch_(self, epoch ):
        print(f"[GPU {self.gpu_id}] Epoch {epoch} | Steps:{len(self.dataset)}")
        self.SE_Model.module.train()
        self.loss_values = []
        for i, data in enumerate(tqdm.tqdm(self.dataset, desc=f"Training Epoch {epoch + 1}")):
            loss = self._run_batch(data,epoch )
            self.loss_values.append(loss)
        self.loss_avg.append(np.mean(self.loss_values))
        Validation_loss=self._validate()
        self.val.append(Validation_loss)


        self.scheduler.step()
        

    def load_from_checkpoint(self, path):
        checkpoint = torch.load(path,weights_only=False)
        self.SE_Model.module.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        return start_epoch


    def train(self, max_epochs, start_epoch ):
        os.makedirs(f"{self.NAME}_Audio", exist_ok=True)

        for epoch in range(start_epoch, max_epochs):
            self._run_epoch_(epoch  )
            if (epoch % self.gen_every == 0) :
                self._denoise_validation(epoch)
            if (self.gpu_id == 0) and (epoch % self.save_every == 0):
                self._save_checkpoint(epoch)
            print(self.loss_avg[-1])
        if (self.gpu_id == 0):
            self._save_checkpoint(max_epochs)
            self._denoise_validation(max_epochs)
            
            
    
 
    def _save_checkpoint(self, epoch):
        SE_Model_params = {
            'input_dim':self.SE_Model.module.input_dim,
            'dim': self.SE_Model.module.dim,
            'max_seq_len': self.SE_Model.module.max_seq_len,
            'N_layers': len(self.SE_Model.module.noise_transformer.layers),
            'dim_head': int(self.SE_Model.module.dim/self.SE_Model.module.noise_transformer.layers[0].heads),
            'heads':self.SE_Model.module.noise_transformer.layers[0].heads#,
        }
        ckp = self.SE_Model.module.state_dict()
        opt = self.optimizer.state_dict()
        os.makedirs("Checkpoints", exist_ok=True)
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
         device,
         DATA_PATHS,
         SAVE_EVERY: int,
         GEN_EVERY:int,
         NUM_EPOCHS: int,
         BATCH_SIZE: int,
         LEARNING_RATE,
         MAX_LEN ,
         Params,
         checkpoint_path,
         NAME,
         sr
         ):
    
    ddp_setup(rank, world_size)
    max_len=int(sr*(MAX_LEN/63))
    clean_training= librosa.util.find_files( DATA_PATHS[0], ext='wav')
    noisy_training= librosa.util.find_files(  DATA_PATHS[1], ext='wav') 
    dataset=labled_AudioDataset(clean_training, noisy_training, max_len,random_start=True)
    train_dataset=DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, sampler=DistributedSampler(dataset))
    clean_validation= librosa.util.find_files( DATA_PATHS[2], ext='wav') 
    noisy_validation= librosa.util.find_files( DATA_PATHS[3], ext='wav') 
    dataset=labled_AudioDataset(clean_validation, noisy_validation, max_len,random_start=True)
    val_dataset=DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,sampler=DistributedSampler(dataset))
    
    SE_Model, optimizer, scheduler = load_train_objs(device,Params,LEARNING_RATE,checkpoint_path,NUM_EPOCHS)
    
    
    parameter_count=par_count(SE_Model)
    print("Number of Model Parameters :",parameter_count)

    trainer = STFT_NAR_Trainer( SE_Model
                      , rank
                      , train_dataset
                      , val_dataset
                      , optimizer
                      , scheduler
                      , SAVE_EVERY
                      , GEN_EVERY
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
                     
                         device,
                         DATA_PATHS,
                         SAVE_EVERY,
                         GEN_EVERY,
                         NUM_EPOCHS,
                         BATCH_SIZE,
                         LEARNING_RATE,
                         MAX_LEN,
                         Params,
                         checkpoint_path,
                         NAME,
                         sr ), nprocs=world_size)
