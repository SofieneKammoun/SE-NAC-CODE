import torch
import torch.nn.functional as F
from torch import nn 
from einops import rearrange,reduce
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


class D_NAR_Model(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
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
        causal=False,
        Nq=12
    ):
        super().__init__()
        self.num_tokens=num_tokens
        self.max_seq_len = max_seq_len
        self.Nq=Nq
        self.dim = dim
        self.token_embs = nn.ModuleList([nn.Embedding(num_tokens, dim) for _ in range(Nq)])
        
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

        self.output_layer = nn.ModuleList([ 
            nn.Sequential(
                FeedForward(dim=self.dim,mult = 8),
                nn.Linear(self.dim, self.num_tokens)
                )for _ in range(Nq)])
        
        # self.output_layer = nn.ModuleList([
        #     nn.Linear(self.dim, self.num_tokens)
        #         for _ in range(Nq)])
        
    def generate(self,noisy):
        logits= self.forward(noisy_tokens=noisy, tokens=None, return_loss=False )
        sampled = logits.argmax(dim = -1)
        return sampled # B x T x D 

    def forward(self, noisy_tokens, tokens , return_loss = False,gen=False):
        assert noisy_tokens.ndim == 3
        if noisy_tokens.numel() == 0:
            return self.forward_empty(noisy_tokens.shape[0])
        T=noisy_tokens.shape[1]
        device = noisy_tokens.device
        assert T <= (self.max_seq_len + 1), 'spatial dimension is greater than the max_seq_len set'
        token_embeddings = []
        for d in range(self.Nq):
            token_emb = self.token_embs[d]
            depth_ids = noisy_tokens[:, :, d]
            tokens_emb = token_emb(depth_ids)
            token_embeddings.append(tokens_emb)
            
        # Stack the embedded noisy_tokens along the depth dimension
        noisy_tokens = torch.stack(token_embeddings, dim=2)
        
        spatial_pos = self.spatial_pos_emb(torch.arange(T, device = device))
        # time_tokens reduced along depth dimension
        time_tokens = reduce(noisy_tokens, 'b t d f -> b t f', 'sum') + spatial_pos

         # spatial noisy_tokens is noisy_tokens with depth pos reduced along depth dimension + spatial positions
        
        
        time_tokens = self.noise_transformer(time_tokens)
        # logits = self.to_logits(time_tokens)
        output_logits=[]
        for d in range(self.Nq):
            out_logit = self.output_layer[d](time_tokens)
            output_logits.append(out_logit)
            
        logits=torch.stack(output_logits, dim=2)
        if not return_loss :
           
            return logits
        preds = rearrange(logits, 'b ... c -> b c (...)')
        labels = rearrange( tokens, 'b t d -> b (t d)')
        labels = labels.long()
        
        loss = F.cross_entropy(preds, labels)
        return loss


