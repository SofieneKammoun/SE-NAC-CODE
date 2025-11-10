#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 13:59:29 2025

@author: root
"""


import os

import torch
import numpy as np
import tqdm
import soundfile as sf
from einops import rearrange
from torch.utils.data import Dataset 
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group 
from pystoi import stoi


def par_count(Model):
    parcount=0
    for p in Model.parameters():
        nn=1
        for s in list(p.size()):
            nn = nn*s
        parcount += nn
    return parcount



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


def cosine_decay(epoch, warmup_epochs, total_epochs):
    if epoch < warmup_epochs:
        return min((epoch + 1) / warmup_epochs, 1.0)  # Warmup phase (Linear)
    else:
        cosine_epoch = epoch - warmup_epochs
        return 0.5 * (1 + np.cos(np.pi * cosine_epoch / (total_epochs - warmup_epochs))) 



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


class Trainer:
    def __init__(self, DAC_Model, SE_Model, gpu_id, dataset,Val_dataset,optimizer, scheduler, save_every,gen_every, Nq,sr,NAME):
        
        """
        Trainer class.

        Args:
            DAC_Model: The DAC model used for training.
            SE_Model: The RQ-Transformer model.
            gpu_id: The GPU device ID to use (rank).
            dataset:     Training dataset.
            Val_dataset: Validation dataset.
            optimizer: Optimizer for model training.
            scheduler: Learning rate scheduler.
            save_every: Interval (in epochs) to save model checkpoints.
            gen_every: Interval (in epochs) to generate samples.
            Nq: Number of quantization levels.
            sr: Sample rate for audio processing.
            NAME: Name of the training experiment.

        Initializes distributed data parallel (DDP) models
        """
        self.DAC_Model = DAC_Model.to(gpu_id)
        self.SE_Model = SE_Model.to(gpu_id)
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
        self.DAC_Model = DDP(DAC_Model, device_ids=[gpu_id])
        self.SE_Model = DDP(SE_Model, device_ids=[gpu_id],find_unused_parameters=True)
        self.NAME=NAME


    def Encode(self,x):
        x =self.DAC_Model.module.preprocess(x,self.sr)
        return self.DAC_Model.module.encoder(x) # : (B x D x T)


    def Tokenize(self,x):
        z_e = self.Encode(x)
        T = z_e.shape[2]
        B = x.shape[0]
        z = torch.empty((self.Nq, B, T), device=self.gpu_id).int()
        for i ,layer in enumerate(self.DAC_Model.module.quantizer.quantizers[:self.Nq]):
            z_in = layer.in_proj(z_e)  # z_in : (B x d x T)
            z_q, indices = layer.decode_latents(z_in)# z_q : (B x D x T) ## indices: (B x T)
            z[i] = indices
            z_q = layer.out_proj(z_q)
            z_e = z_e - z_q
        z = rearrange(z, "n b t -> b t n")
        
        return z  # z: (B x T x N)


    def Detokenize(self,z):
        z_q = 0.0
        for i in range(self.Nq):
            z_p_i = self.DAC_Model.module.quantizer.quantizers[i].decode_code(z[:, i, :])
            z_q_i = self.DAC_Model.module.quantizer.quantizers[i].out_proj(z_p_i)
            z_q = z_q + z_q_i
        return z_q  


    def Decode(self,z_q):
        return self.DAC_Model.module.decoder(z_q)



    def process_batch_train_audio(self, batch):
        
        x=self.Tokenize(batch[0].to(self.gpu_id))
        y=self.Tokenize(batch[1].to(self.gpu_id))
        
        loss = self.SE_Model(noisy_tokens=y, tokens=x,return_loss=True)
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
        Logs reconstructed, noisy, and clean signals for experiment tracking.
        """
        self.SE_Model.module.eval()
        with torch.no_grad():
            SI_SDR = []
            ESTOI =[]
            k=np.random.randint(self.val_dataset.__len__())

            for i , data in enumerate(tqdm.tqdm(self.val_dataset, desc="Validation : Denoising")):
                xc = data[0].to(self.gpu_id)
                xn=self.Tokenize(data[1].to(self.gpu_id))
                y_,y_gen=self.denoise(xn)
                x_n=data[1].to(self.gpu_id)
                
                y = self.Decode(y_)
                y = y.detach().cpu().numpy().squeeze()
                xc = xc.detach().cpu().numpy().squeeze()
                x_n = x_n.detach().cpu().numpy().squeeze()
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


    def denoise(self, y):
        """
       Denoises a given input using the trained model.

       Args:
           y: Input noisy tokenized speech.

       Returns:
           Detokenized latent representation z_q and generated quantized tokens z_gen.
       """
        seq_lenght=y.shape[1]
        z_gen = self.SE_Model.module.generate(noisy=y,
                                           filter_thres=1,
                                           spatial_seq_len=seq_lenght,
                                           default_batch_size= y.shape[0])
        z_gen = rearrange(z_gen,"b t n -> b n t")
        z_q = self.Detokenize(z_gen)
        
        return z_q , z_gen


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
        

    def _save_checkpoint(self, epoch):
        SE_Model_params = {
            'num_tokens': self.SE_Model.module.token_embs[0].num_embeddings,
            'dim': self.SE_Model.module.dim,
            'noise_dim': self.SE_Model.module.noise_dim,
            'max_spatial_seq_len': self.SE_Model.module.max_spatial_seq_len,
            'depth_seq_len': self.SE_Model.module.depth_seq_len,
            'noise_depth': self.SE_Model.module.noise_depth,
            'spatial_layers': len(self.SE_Model.module.spatial_transformer.layers),
            'depth_layers': len(self.SE_Model.module.depth_transformer.layers),
            'dim_head': int(self.SE_Model.module.dim/self.SE_Model.module.depth_transformer.layers[0][0].heads),
            'heads':self.SE_Model.module.depth_transformer.layers[0][0].heads
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
