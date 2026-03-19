import torch
from torch import nn 
from einops import  repeat
from Models.Conformer import Conformer


def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)


def remainder_to_mult(num, mult):
    return (mult - num % mult) % mult

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / temperature) + gumbel_noise(t)).argmax(dim = dim)

def top_k(logits, thres = 0.5):
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs


def FeedForward(*, dim, mult = 4, dropout = 0.):
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, dim * mult),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dim * mult, dim)
    )

class C_NAR_Model(nn.Module):
    def __init__(
        self,
        *,
        input_dim,
        dim,
        max_seq_len,
        N_layers,
        dim_head = 64,
        heads = 8,
        attn_dropout = 0.,
        ff_mult = 4,
        ff_dropout = 0.,
        pad_id = 0,
        conv_kernel_size=10,
        causal=False
    ):
        super().__init__()
        
        self.max_seq_len = max_seq_len
        
        self.dim = dim
        self.input_dim=input_dim
        # PE_t
        self.spatial_pos_emb = nn.Embedding(max_seq_len + 1,self.dim) # account for a boundary case
        self.causal=causal
        self.noise_transformer = Conformer(
            dim = self.dim,
            layers = N_layers,
            dim_head = dim_head,
            heads = heads,
            conv_kernel_size = conv_kernel_size,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            ff_mult = ff_mult,
            conv_causal=self.causal
        )

        self.input_layer  = nn.Linear(self.input_dim, self.dim)
        self.output_layer = nn.Linear(self.dim, self.input_dim)
    def forward_empty(self, batch_size,noisy):
        spatial_tokens = repeat(self.spatial_start_token, 'd -> b 1 d', b = batch_size)
        logits = self.spatial_transformer(spatial_tokens)

        return logits

    def forward(self, embeds, clean_embeds=None , return_loss = False,gen=False):
        assert embeds.ndim == 3
        if embeds.numel() == 0:
            return self.forward_empty(embeds.shape[0])
        
        tokens = self.input_layer(embeds)
        
        spatial_tokens = tokens 
        
        
        spatial_tokens = self.noise_transformer(spatial_tokens)
        mask = self.output_layer(spatial_tokens)
        mask = torch.sigmoid(mask)
        masked_embeds = embeds * mask
        if not return_loss :
           
            return masked_embeds#[:, 1:]
        
        labels = clean_embeds
        loss = torch.nn.functional.mse_loss(masked_embeds, labels)
        return loss


