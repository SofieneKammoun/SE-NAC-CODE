#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 28 16:03:24 2025

@author: root
"""


import os

import torch
import numpy as np
import librosa
import tqdm
import soundfile as sf
from einops import rearrange
from torch.utils.data import Dataset, DataLoader

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import pandas as pd
from pystoi import stoi
import dac

DATA_PATHS = [
    "Path/to/train-360/s1",
    "Path/to/train-360/mix_single",
    "Path/to/dev/s1",
    "Path/to/dev/mix_single",
    ] 

model_file ="weights_16khz_8kbps_0.0.5.pth"

NUM_EPOCHS =300
BATCH_SIZE =64
SAVE_EVERY=50
GEN_EVERY=25
LEARNING_RATE =0.0001 * (BATCH_SIZE/256)
checkpoint_path=None#"Checkpoints/DAC_Encoder_ckpt_100.pt"

sr=16000
MAX_LEN =50
Nq=12
NAME="DAC_Encoder_Cont"
device='cuda'

def ddp_setup(rank: int, world_size: int):
  """
  Args:
      rank: Unique identifier of each process
     world_size: Total number of processes
  """
  os.environ["MASTER_ADDR"] = "localhost"
  os.environ["MASTER_PORT"] = "12355"
  torch.cuda.set_device(rank)
  init_process_group(backend="nccl", rank=rank, world_size=world_size)

def compute_sisdr(estimate, reference):

    eps = np.finfo(estimate.dtype).eps
    alpha = (np.sum(estimate*reference) + eps) / (np.sum(np.abs(reference)**2) + eps)
    sisdr = 10*np.log10((np.sum(np.abs(alpha*reference)**2) + eps)/
                        (np.sum(np.abs(alpha*reference - estimate)**2) + eps))
    return sisdr
class labled_AudioDataset(Dataset):
    def __init__(self, clean_list,noisy_list, max_len, random_start=True):
        self.clean_list = clean_list
        self.noisy_list = noisy_list
        self.max_len = max_len
        self.random_start = random_start
    def __len__(self):
        return len(self.clean_list)

    def __getitem__(self, idx):
        wav_file = self.clean_list[idx]
        wav_noisy = self.noisy_list[idx]
        with sf.SoundFile(wav_file) as f:
            total_frames = len(f)
        required_frames = self.max_len
        if total_frames < required_frames:
            speech, sr = sf.read(wav_file)
            speech_noisy, sr = sf.read(wav_noisy)
            pad_size = required_frames - total_frames
            speech = np.pad(speech, (0, pad_size), 'constant')
            speech_noisy = np.pad(speech_noisy, (0, pad_size), 'constant')
        else:
            if self.random_start:
                rand_start = torch.randint(0, total_frames - required_frames + 1, (1,)).item()
            else:
                rand_start=0
            rand_stop = rand_start + required_frames
            speech, sr = sf.read(wav_file, start=rand_start, stop=rand_stop)
            speech_noisy, sr = sf.read(wav_noisy, start=rand_start, stop=rand_stop)
        speech = speech[np.newaxis, np.newaxis, :]
        speech_noisy = speech_noisy[np.newaxis, np.newaxis, :]
        return torch.from_numpy(speech).float().squeeze(0),torch.from_numpy(speech_noisy).float().squeeze(0)

def cosine_decay(epoch, warmup_epochs, total_epochs):
    if epoch < warmup_epochs:
        return min((epoch + 1) / warmup_epochs, 1.0)  # Warmup phase (Linear)
        # return 1.0 
    else:
        cosine_epoch = epoch - warmup_epochs
        return 0.5 * (1 + np.cos(np.pi * cosine_epoch / (total_epochs - warmup_epochs)))  # Cosine decay


def load_train_objs( model_file, device,MAX_LEN ,Nq,LEARNING_RATE,checkpoint_path,NUM_EPOCHS):
    device='cuda'
    model_c = dac.DAC.load(model_file)
    model_c.encoder.to(device)
    model_c.quantizer.to(device)
    model = dac.DAC.load(model_file)
    model.encoder.to(device)
    model.quantizer.to(device)
    
    Time = MAX_LEN
    print('Time =' , Time, '\n nq =', Nq)
    
    optimizer = torch.optim.AdamW(model.encoder.parameters(),
                                  lr= LEARNING_RATE,
                                  betas=(0.9, 0.95),
                                  weight_decay=0.05)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                     lr_lambda=lambda epoch: cosine_decay(epoch,20, NUM_EPOCHS))

    return model,model_c, optimizer, lr_scheduler
class Trainer:
    def __init__(self, model,model_c ,  gpu_id, dataset,Val_dataset,optimizer, scheduler, save_every,gen_every, Nq,sr,NAME):
        
        """
        Trainer class.

        Args:
            model: The Funcodec model used for training.
            gpu_id: The GPU device ID to use (rank).
            dataset:     Training dataset.
            Val_dataset: Validation dataset.
            optimizer: Optimizer for model training.
            scheduler: Learning rate scheduler.
            save_every: Interval (in epochs) to save model checkpoints.
            gen_every: Interval (in epochs) to generate samples.
            Nq: Number of quantization levels.
            sr: Sample rate for audio processing.
            NAME: Name of the training .

        Initializes distributed data parallel (DDP) models and prepares codebooks for quantization.
        """
        self.model = model.to(gpu_id)
        self.model_c = model_c.to(gpu_id)
        self.gpu_id = gpu_id
        self.dataset = dataset
        self.val_dataset = Val_dataset
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_every = save_every
        self.gen_every = gen_every
        self.Nq = Nq
        self.sr=sr
        self.total_steps = 0
        self.loss_values = []
        self.loss_avg = []
        self.val = []
        self.lr = []
        self.model = DDP(model, device_ids=[gpu_id])
        self.model_c = DDP(model_c, device_ids=[gpu_id])
        self.NAME=NAME
        
    def decode_latents(self,layer, latents):
        
        encodings = rearrange(latents, "b d t -> (b t) d")
        
        with torch.no_grad():
            codebook = layer.codebook.weight  # codebook: (N x D)


        # Compute euclidean distance with codebook
        dist =(
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )
        
        indices = rearrange((-dist).max(1)[1], "(b t) -> b t", b=latents.size(0))
        z_q = layer.decode_code(indices)
        return z_q, indices
    
    
    def Tokenize(self,model,x):
        with torch.no_grad():
            x =model.module.preprocess(x,self.sr)
        z_e = model.module.encoder(x)# z_e : (B x D x T)
        if z_e.requires_grad==True :
            z_e.retain_grad()
        T = z_e.shape[2]
        B = z_e.shape[0]
        z_out=torch.zeros_like(z_e)
        z = torch.empty((self.Nq, B, T), device=self.gpu_id).int()
        
        for i ,layer in enumerate(model.module.quantizer.quantizers[:self.Nq]):
            z_in = layer.in_proj(z_e)  # z_in : (B x d x T)
            z_q, indices = self.decode_latents(layer,z_in)# z_q : (B x D x T) ## indices: (B x T)
            z_q = layer.out_proj(z_q)
            z_out=z_out+z_q
            z_e = z_e - z_q
            z[i] = indices

        z = rearrange(z, "n b t -> b t n")
        return z_out , z 

    def process_batch_train_audio(self, batch):
        with torch.no_grad():
            x =self.model_c.module.preprocess(batch[0].to(self.gpu_id).detach(),self.sr)
            x= self.model_c.module.encoder(x)# z_e : (B x D x T)
        y = self.model.module.preprocess(batch[1].to(self.gpu_id),self.sr)
        y = self.model.module.encoder(y)# z_e : (B x D x T)
        loss= torch.nn.functional.mse_loss(y, x.detach())
        return loss

    def _run_batch(self, batch,epoch ):
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
        
        self.model.module.eval()
        with torch.no_grad():
            SI_SDR = []
            ESTOI =[]
            k=np.random.randint(self.val_dataset.__len__())
            for i , data in enumerate(tqdm.tqdm(self.val_dataset, desc="Validation : Denoising")):
                y_= self.Tokenize(self.model, data[1].to(self.gpu_id))[0]
                y = self.model.module.decoder(y_)
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
        self.model.eval()
        with torch.no_grad():
            loss_values = []
            for i , data in enumerate(tqdm.tqdm(self.val_dataset, desc="Validation :")):
                    loss = self.process_batch_train_audio(data)
                    loss_values.append(loss.item())
        
        return np.mean(loss_values)
    def _run_epoch_(self, epoch):
        print(f"[GPU {self.gpu_id}] Epoch {epoch} | Steps:{len(self.dataset)}")
        self.model.module.train()
        self.model_c.module.eval()
        self.loss_values = []
        for i, data in enumerate(tqdm.tqdm(self.dataset, desc=f"Training Epoch {epoch + 1}")):
            loss = self._run_batch(data,epoch)
            self.loss_values.append(loss)
        self.loss_avg.append(np.mean(self.loss_values))
        Validation_loss=self._validate()
        self.val.append(Validation_loss)
        self.scheduler.step()
        

    def _save_checkpoint(self, epoch):
        
        ckp = self.model.module.state_dict()
        opt = self.optimizer.state_dict()
        PATH = f"Checkpoints/{self.NAME}_ckpt_{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict':ckp,
            'optimizer_state_dict': opt
            }, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")
    def load_from_checkpoint(self, path):
        checkpoint = torch.load(path,weights_only=False)
        self.model.module.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        return start_epoch
    def train(self, max_epochs, start_epoch):
        for epoch in range(start_epoch, max_epochs):
            if (epoch % self.gen_every == 0) :
                self._denoise_validation(epoch)
            self._run_epoch_(epoch)
            # if (self.gpu_id == 0) and (epoch % self.save_every == 0):
            #     self._save_checkpoint(epoch)
        if (self.gpu_id == 0):
            self._save_checkpoint(max_epochs)
            # self._denoise_validation(max_epochs)
def main(rank: int, world_size: int, model_file,device, SAVE_EVERY: int,GEN_EVERY:int, NUM_EPOCHS: int,BATCH_SIZE: int,LEARNING_RATE  ,Nq,MAX_LEN ,checkpoint_path,NAME,sr):
    ddp_setup(rank, world_size)
    max_len=int(sr*(MAX_LEN/50))
    clean_training= librosa.util.find_files( DATA_PATHS[0], ext='wav') 
    noisy_training= librosa.util.find_files(  DATA_PATHS[1], ext='wav') 
    dataset=labled_AudioDataset(clean_training, noisy_training, max_len,random_start=False)
    train_dataset=DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, sampler=DistributedSampler(dataset))
    clean_validation= librosa.util.find_files( DATA_PATHS[2], ext='wav') 
    noisy_validation= librosa.util.find_files( DATA_PATHS[3], ext='wav') 
    dataset=labled_AudioDataset(clean_validation, noisy_validation, max_len,random_start=False)
    val_dataset=DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,sampler=DistributedSampler(dataset))

    model ,model_c , optimizer, scheduler = load_train_objs(model_file,device,MAX_LEN ,Nq,LEARNING_RATE,checkpoint_path,NUM_EPOCHS)
    
    
    parcount=0
    for p in model.parameters():
        nn=1
        for s in list(p.size()):
            nn = nn*s
        parcount += nn
    print(parcount)
    
    trainer = Trainer(  model
                      , model_c
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
                         model_file,
                         device,
                         SAVE_EVERY,
                         GEN_EVERY,
                         NUM_EPOCHS,
                         BATCH_SIZE,
                         LEARNING_RATE,
                         Nq,
                         MAX_LEN,
                         checkpoint_path,
                         NAME,
                         sr ), nprocs=world_size)
