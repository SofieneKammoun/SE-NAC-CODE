import torch
import torch.nn.functional as F
from torch import nn 
from tqdm import tqdm
from einops import rearrange, reduce, repeat
from Models.Conformer import Conformer
from Models.Transformer import Transformer


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


class D_AR_Model(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        noise_dim,
        max_spatial_seq_len,
        depth_seq_len,
        noise_depth,
        spatial_layers,
        depth_layers,

        dim_head = 64,
        heads = 8,
        attn_dropout = 0.,
        ff_mult = 4,
        ff_dropout = 0.,
        pad_id = 0,
        conv_kernel_size=10
    ):
        super().__init__()
        
        self.max_spatial_seq_len = max_spatial_seq_len
        self.depth_seq_len = depth_seq_len
        self.noise_depth = noise_depth
        
        self.dim = dim
        self.noise_dim = noise_dim
        
        # an embedding layer for each quantization stage
        self.token_embs = nn.ModuleList([nn.Embedding(num_tokens, dim) for _ in range(depth_seq_len)])
        self.noise_embs = nn.ModuleList([nn.Embedding(num_tokens, noise_dim) for _ in range(noise_depth)])
        # u_0
        self.spatial_start_token = nn.Parameter(torch.randn(self.dim))
        # PE_t
        self.spatial_pos_emb = nn.Embedding(max_spatial_seq_len + 1,self.dim) # account for a boundary case
        self.noise_pos_emb = nn.Embedding(max_spatial_seq_len, dim) #  diffrent positional embedding layer for noisy inputs
        
        # PE_d
        self.depth_pos_emb = nn.Embedding(depth_seq_len,self.dim)

        self.noise_transformer = Conformer(
            dim = self.noise_dim,
            layers = spatial_layers,
            dim_head = dim_head,
            heads = heads,
            conv_kernel_size = conv_kernel_size,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            ff_mult = ff_mult,
            conv_causal=False
        )

        self.spatial_transformer = Conformer(
            dim = self.noise_dim,
            layers = spatial_layers,
            dim_head = dim_head,
            heads = heads,
            conv_kernel_size = conv_kernel_size,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            ff_mult = ff_mult,
            conv_causal=True
        )


        self.depth_transformer = Transformer(
            dim = self.dim,
            layers = depth_layers,
            dim_head = dim_head,
            heads = heads,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            ff_mult = ff_mult
        )

        self.to_logits = nn.Linear(self.dim, num_tokens)
        self.pad_id = pad_id
    def generate(self, prime=None , noisy =None, filter_thres=1, temperature=0.5, default_batch_size=1, spatial_seq_len=None):
        if spatial_seq_len is None:
            spatial_seq_len = self.max_spatial_seq_len
        max_seq_len=self.depth_seq_len *self.max_spatial_seq_len
        device = next(self.parameters()).device
        if not exists(prime):
            prime = torch.empty((default_batch_size, 0), dtype = torch.long, device = device)
        full_seq = torch.empty((default_batch_size, 0), dtype = torch.long, device = device)
        seq = prime
        
        noisy_embeddings = []
        noisy_depth = noisy.shape[2]
        noisy_chunck=noisy[:,:self.max_spatial_seq_len]
        for d in range(noisy_depth):
            noisy_emb = self.noise_embs[d]
            depth_noisy =noisy_chunck[:, :, d]
            noisy_tokens = noisy_emb(depth_noisy)
            noisy_embeddings.append(noisy_tokens)
        noisy_tokens = torch.stack(noisy_embeddings, dim=2)
        noisy_pos = self.noise_pos_emb(torch.arange(noisy_tokens.shape[1], device = device))
        noisy_tokens = reduce(noisy_tokens, 'b s d f -> b s f', 'sum') + noisy_pos
        noisy_tokens = self.noise_transformer(noisy_tokens)
        for i in tqdm(range(spatial_seq_len-(seq.shape[-1]//self.depth_seq_len))):# -(seq.shape[-1]//self.depth_seq_len)
            for j in range(self.depth_seq_len):
                # Truncate seq if necessary
                truncated_noisy=noisy_tokens[:,:self.max_spatial_seq_len]
                if seq.shape[-1] >= max_seq_len :
                    full_seq=torch.cat((full_seq,seq),dim =1)
                    seq=torch.empty((default_batch_size, 0), dtype = torch.long, device = device)
                    noisy=noisy[:,self.max_spatial_seq_len:]
                    noisy_embeddings = []
                    noisy_chunck=noisy[:,:self.max_spatial_seq_len]
                    for d in range(noisy_depth):
                        noisy_emb = self.noise_embs[d]
                        depth_noisy =noisy_chunck[:, :, d]
                        noisy_tokens = noisy_emb(depth_noisy)
                        noisy_embeddings.append(noisy_tokens)
                    noisy_tokens = torch.stack(noisy_embeddings, dim=2)
                    noisy_pos = self.noise_pos_emb(torch.arange(noisy_tokens.shape[1], device = device))
                    noisy_tokens = reduce(noisy_tokens, 'b s d f -> b s f', 'sum') + noisy_pos
                    noisy_tokens = self.noise_transformer(noisy_tokens)
                truncated_seq = seq
                    
                logits = self.infere(truncated_seq,gen=True,noisy_tokens=truncated_noisy)[:, -1]
                if j ==0:
                    logits = top_k(logits, thres = filter_thres)
                else:
                    logits = top_k(logits, thres =1)
                sampled = gumbel_sample(logits, dim = -1, temperature =temperature)
                seq = torch.cat((seq, rearrange(sampled, 'b -> b 1')), dim = -1)
        seq=torch.cat((full_seq,seq),dim =1)
        
        return rearrange(seq, 'b (s d) -> b s d', d = self.depth_seq_len)

    def forward_empty(self, batch_size,noisy):
        spatial_tokens = repeat(self.spatial_start_token, 'd -> b 1 d', b = batch_size)
        spatial_tokens = self.spatial_transformer(spatial_tokens)
        depth_tokens = torch.cat((noisy,spatial_tokens), dim = -2)

        depth_tokens = self.depth_transformer(depth_tokens)[:,1:]
        
        logits=self.to_logits(depth_tokens)
        
        return logits

    def forward(self, tokens, return_loss = False, noisy_tokens=None,gen=False):
        
        ids=tokens
        noisy=noisy_tokens
        assert ids.ndim in {2, 3}
        flattened_dim = ids.ndim == 2

        if ids.numel() == 0:
            return self.forward_empty(ids.shape[0],noisy)
       

        if flattened_dim:
            seq_len = ids.shape[-1]
            padding = remainder_to_mult(seq_len, self.depth_seq_len)
            ids = F.pad(ids, (0, padding), value = self.pad_id)
            ids = rearrange(ids, 'b (s d) -> b s d', d = self.depth_seq_len)
        else:
            seq_len = ids.shape[1] * ids.shape[2]
        noisy_len = noisy.shape[1]
        noisy_depth = noisy.shape[2]
        b, space, depth, device = *ids.shape, ids.device
        assert space <= (self.max_spatial_seq_len + 1), 'spatial dimension is greater than the max_spatial_seq_len set'
        assert depth == self.depth_seq_len, 'depth dimension must be equal to depth_seq_len'
        assert noisy_depth == self.noise_depth, 'depth dimension must be equal to noise_depth'
        # get token embeddings
        token_embeddings = []
        noisy_embeddings = []
        for d in range(depth):
            # Select the appropriate embedding layer for the current depth dimension
            token_emb = self.token_embs[d]
            # Get the ids for the current depth dimension
            depth_ids = ids[:, :, d]
            # Embed the tokens
            tokens = token_emb(depth_ids)
            token_embeddings.append(tokens)
        for d in range(noisy_depth):
            noisy_emb = self.noise_embs[d]
            depth_noisy = noisy[:, :, d]
            noisy_tokens = noisy_emb(depth_noisy)
            noisy_embeddings.append(noisy_tokens)
            
            
        # Stack the embedded tokens along the depth dimension
        tokens = torch.stack(token_embeddings, dim=2)
        noisy_tokens = torch.stack(noisy_embeddings, dim=2)
        
        spatial_pos = self.spatial_pos_emb(torch.arange(space    , device = device))
        noisy_pos   = self.noise_pos_emb(  torch.arange(noisy_len, device = device))
        
        depth_pos = self.depth_pos_emb(torch.arange(depth, device = device))
        tokens_with_depth_pos = tokens + depth_pos

         # spatial tokens is tokens with depth pos reduced along depth dimension + spatial positions
        spatial_tokens = reduce(tokens_with_depth_pos, 'b s d f -> b s f', 'sum') + spatial_pos 
        noisy_tokens = reduce(noisy_tokens, 'b s d f -> b s f', 'sum') + noisy_pos 
        
        spatial_tokens = torch.cat((
            repeat(self.spatial_start_token, 'f -> b 1 f', b = b),
            spatial_tokens
        ), dim = -2)
        
        spatial_tokens = self.spatial_transformer(spatial_tokens)
        noisy_tokens = self.noise_transformer(noisy_tokens)


        spatial_tokens = rearrange(spatial_tokens, 'b s f -> b s 1 f')
        noisy_tokens   = rearrange(noisy_tokens  , 'b s f -> b s 1 f')

        noisy_tokens = F.pad(noisy_tokens, (0, 0, 0, 0, 0, 1), value = 0.)
        tokens_with_depth_pos = F.pad(tokens_with_depth_pos, (0, 0, 0, 0, 0, 1), value = 0.)
        noisy_tokens=noisy_tokens[:,:space+1]
        depth_tokens = torch.cat((noisy_tokens,spatial_tokens, tokens_with_depth_pos), dim = -2)

        depth_tokens = rearrange(depth_tokens, '... d f -> (...) d f')

        depth_tokens = self.depth_transformer(depth_tokens)[:,1:]
        depth_tokens = rearrange(depth_tokens, '(b s) d f -> b s d f', b = b)

        logits = self.to_logits(depth_tokens)
        logits = rearrange(logits, 'b ... f -> b (...) f')
        
        logits = logits[:, :(seq_len + 1)]

        if not return_loss :
            if not gen : 
                return logits[:, :-1]
            return logits[:, 1:]

        logits = logits[:, :-1]
        
        preds = rearrange(logits, 'b ... c -> b c (...)')
        labels = rearrange(ids, 'b s d -> b (s d)')
        labels = labels.long()
        
        loss = F.cross_entropy(preds, labels, ignore_index = self.pad_id)
        return loss


    def infere(self, ids, return_loss = False, noisy_tokens=None,gen=False):
        assert ids.ndim in {2, 3}
        flattened_dim = ids.ndim == 2

        if ids.numel() == 0:
            return self.forward_empty(ids.shape[0],noisy_tokens)
       

        if flattened_dim:
            seq_len = ids.shape[-1]
            padding = remainder_to_mult(seq_len, self.depth_seq_len)
            ids = F.pad(ids, (0, padding), value = self.pad_id)
            ids = rearrange(ids, 'b (s d) -> b s d', d = self.depth_seq_len)
        else:
            seq_len = ids.shape[1] * ids.shape[2]
        b, space, depth, device = *ids.shape, ids.device
        assert space <= (self.max_spatial_seq_len + 1), 'spatial dimension is greater than the max_spatial_seq_len set'
        assert depth == self.depth_seq_len, 'depth dimension must be equal to depth_seq_len'
        # get token embeddings
        token_embeddings = []
        for d in range(depth):
            # Select the appropriate embedding layer for the current depth dimension
            token_emb = self.token_embs[d]
            # Get the ids for the current depth dimension
            depth_ids = ids[:, :, d]
            # Embed the tokens
            tokens = token_emb(depth_ids)
            token_embeddings.append(tokens)
            
        # Stack the embedded tokens along the depth dimension
        tokens = torch.stack(token_embeddings, dim=2)
        
        spatial_pos = self.spatial_pos_emb(torch.arange(space    , device = device))
        
        depth_pos = self.depth_pos_emb(torch.arange(depth, device = device))
        tokens_with_depth_pos = tokens + depth_pos

         # spatial tokens is tokens with depth pos reduced along depth dimension + spatial positions
        spatial_tokens = reduce(tokens_with_depth_pos, 'b s d f -> b s f', 'sum') + spatial_pos 
        
        spatial_tokens = torch.cat((
            repeat(self.spatial_start_token, 'f -> b 1 f', b = b),
            spatial_tokens
        ), dim = -2)
        
        spatial_tokens = self.spatial_transformer(spatial_tokens)


        spatial_tokens = rearrange(spatial_tokens, 'b s f -> b s 1 f')
        noisy_tokens   = rearrange(noisy_tokens  , 'b s f -> b s 1 f')

        noisy_tokens = F.pad(noisy_tokens, (0, 0, 0, 0, 0, 1), value = 0.)
        tokens_with_depth_pos = F.pad(tokens_with_depth_pos, (0, 0, 0, 0, 0, 1), value = 0.)
        noisy_tokens=noisy_tokens[:,:space+1]
        depth_tokens = torch.cat((noisy_tokens,spatial_tokens, tokens_with_depth_pos), dim = -2)

        depth_tokens = rearrange(depth_tokens, '... d f -> (...) d f')

        depth_tokens = self.depth_transformer(depth_tokens)[:,1:]
        depth_tokens = rearrange(depth_tokens, '(b s) d f -> b s d f', b = b)

        logits = self.to_logits(depth_tokens)
        logits = rearrange(logits, 'b ... f -> b (...) f')
        
        logits = logits[:, :(seq_len + 1)]

        if not return_loss :
            if not gen : 
                return logits[:, :-1]
            return logits[:, 1:]

        logits = logits[:, :-1]
        
        preds = rearrange(logits, 'b ... c -> b c (...)')
        labels = rearrange(ids, 'b s d -> b (s d)')
        labels = labels.long()
        
        loss = F.cross_entropy(preds, labels, ignore_index = self.pad_id)
        return loss
