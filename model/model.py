import torch
import torch.nn as nn
from model import attention

class GPTConfig:
    embed_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    additive = False

    def __init__(self, vocab_size, args_dict):

        self.vocab_size = vocab_size
        self.__dict__.update(args_dict)


class GPT1Config(GPTConfig):
    n_layer = 12
    n_head = 12
    n_embed = 768


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.ln1 = nn.LayerNorm(config.n_embed)
        self.ln2 = nn.LayerNorm(config.n_embed)
        self.attn = attention.CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embed, 4 * config.n_embed),
            nn.GELU(),
            nn.Linear(4 * config.n_embed, config.n_embed),
            nn.Dropout(config.resid_pdrop)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()

        # input embedding stem
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embed)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embed))
        self.drop = nn.Dropout(config.embed_pdrop)

        # transformer network
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embed)
        self.head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
        self.block_size = config.block_size

        self.model_config = config

        print('Number of parameters: {}'.format(sum(p.numel() for p in self.parameters())))

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):
        b, t = idx.size()

        token_embeddings = self.tok_emb(idx)
        position_embeddings = self.pos_emb[:, :t, :]
        x = self.drop(token_embeddings + position_embeddings)

        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                input=logits.view(-1, logits.size(-1)), 
                target=targets.view(-1), 
                ignore_index=0
            )

        return logits, loss