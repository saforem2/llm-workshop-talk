# Creating Small(-ish) LLMs
Sam Foreman
2024-02-12

# Creating Small(-ish) LLMs[^1]

<div>

</div>

# Emergent Abilities

<div width="66%" style="text-align: center;">

<img src="https://github.com/saforem2/llm-lunch-talk/blob/main/docs/assets/emergent-abilities.gif?raw=true" height="75%" />

[Emergent abilities of Large Language
Models](https://arxiv.org/abs/2206.07682) Yao et al. (2023)

</div>

# Training LLMs

<div>

</div>

# Life-Cycle of the LLM

<div>

</div>

# Forward Pass

<video data-autoplay src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/assisted-generation/gif_1_1080p.mov">
</video>

# Generating Text

<video data-autoplay src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/assisted-generation/gif_2_1080p.mov">
</video>

# Life-Cycle of the LLM: Pre-training

![](https://jalammar.github.io/images/gpt3/03-gpt3-training-step-back-prop.gif)

# Life-Cycle of the LLM: Fine-Tuning

![](https://jalammar.github.io/images/gpt3/10-gpt3-fine-tuning.gif)

# Assistant Models

<span class="preview-image"
style="text-align:center; margin-left:auto; margin-right: auto;">![](https://github.com/saforem2/LLM-tutorial/blob/main/docs/assets/jailbreak.jpeg?raw=true)</span>

# [`saforem2/wordplay` 🎮💬](https://github.com/saforem2/wordplay)

<!-- - [ `saforem2/wordplay`](https://github.com/saforem2/wordplay) -->

- Fork of Andrej Karpathy’s `nanoGPT`

![](https://github.com/saforem2/nanoGPT/raw/master/assets/nanogpt.jpg)

# Install

``` bash
git clone https://github.com/saforem2/wordplay
cd wordplay
mkdir -p venv
python3 -m venv venv --system-site-packages
source venv/bin/activate
python3 -m pip install -e .
python3 -c 'import wordplay; print(wordplay.__file__)'
# ./wordplay/src/wordplay/__init__.py
```

# Dependencies

- [`transformers`](https://github.com/huggingface/transformers) for
  transformers (to load `GPT-2` checkpoints)
- [`datasets`](https://github.com/huggingface/datasets) for datasets (if
  you want to use OpenWebText)
- [`tiktoken`](https://github.com/openai/tiktoken) for OpenAI’s fast BPE
  code
- [`wandb`](https://wandb.ai) for optional logging
- [`tqdm`](https://github.com/tqdm/tqdm) for progress bars

# Quick Start

- We start with training a character-level GPT on the works of
  Shakespeare.

  1.  Downloading the data (~ 1MB) file
  2.  Convert raw text to one large stream of integers

  ``` bash
  python3 data/shakespeare_char/prepare.py
  ```

  This will create `data/shakespeare_char/{train.bin, val.bin}`.

# Model [ `model.py`](https://github.com/saforem2/wordplay/blob/master/src/wordplay/model.py)

<div class="panel-tabset"
style="font-size: 0.75em; width: 100%!important; height: 100%!important;">

### `CausalSelfAttention`

``` python

class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTModelConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(
            config.n_embd,
            3 * config.n_embd,
            bias=config.bias
        )
        # output projection
        self.c_proj = nn.Linear(
            config.n_embd,
            config.n_embd,
            bias=config.bias
        )
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in
        # PyTorch >= 2.0
        self.flash = hasattr(
            torch.nn.functional,
            'scaled_dot_product_attention'
        )
        # if self.flash and RANK == 0:
        #     log.warning(
        #         f'Using torch.nn.functional.scaled_dot_product_attention'
        #         '(Flash Attn)'
        #     )
        if not self.flash:
            log.warning(
                "WARNING: using slow attention."
                "Flash Attention requires PyTorch >= 2.0"
            )
            # causal mask to ensure that attention is only applied to the left
            # in the input sequence
            self.register_buffer(
                "bias",
                torch.tril(
                    torch.ones(
                        config.block_size,
                        config.block_size
                    )
                ).view(1, 1, config.block_size, config.block_size)
            )

    def forward(self, x):
        # batch size, sequence length, embedding dimensionality (n_embd)
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head
        # forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # causal self-attention; Self-attend:
        # (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=(self.dropout if self.training else 0),
                is_causal=True
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(
                self.bias[:, :, :T, :T] == 0,  # type:ignore
                float('-inf')
            )
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
```

### `LayerNorm`

``` python
class LayerNorm(nn.Module):
    """
    LayerNorm but with an optional bias.

    (PyTorch doesn't support simply bias=False)
    """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(
            input,
            self.weight.shape,
            self.weight,
            self.bias,
            1e-5
        )
```

### `MLP`

``` python
class MLP(nn.Module):

    def __init__(
            self,
            config: GPTModelConfig,
            activation: str = 'gelu',
    ):
        super().__init__()
        self.c_fc = nn.Linear(
            config.n_embd,
            4 * config.n_embd,
            bias=config.bias
        )
        if activation.lower() in ACTIVATIONS:
            self.act_fn = ACTIVATIONS[activation.lower()]
        else:
            try:
                act_fn = getattr(nn, activation)
                assert callable(act_fn)
                self.act_fn = act_fn()
            except Exception as exc:
                log.error(f'{activation} not yet supported!')
                raise exc
        # self.gelu = nn.GELU()
        self.c_proj = nn.Linear(
            4 * config.n_embd,
            config.n_embd,
            bias=config.bias
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        # x = self.gelu(x)
        x = self.act_fn(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
```

### `Block`

``` python
class Block(nn.Module):

    def __init__(self, config: GPTModelConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
```

### `GPT`

``` python
class GPT(nn.Module):
    def __init__(self, config: GPTModelConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get
        # generated: "UserWarning: functional_call was passed multiple values
        # for tied weights. This behavior is deprecated and will be an error in
        # future versions" not 100% sure what this is, so far seems to be
        # harmless. TODO investigate
        # https://paperswithcode.com/method/weight-tying
        self.transformer.wte.weight = self.lm_head.weight  # type:ignore

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(
                    p,
                    mean=0.0,
                    std=0.02/math.sqrt(2 * config.n_layer)
                )

        # report number of parameters
        log.info("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get
        subtracted.

        The token embeddings would too, except due to the parameter sharing
        these params are actually used as weights in the final layer, so we
        include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()  # type:ignore
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, (
            f"Cannot forward sequence of length {t}, "
            "block size is only {self.config.block_size}"
        )
        pos = torch.arange(
            0,
            t,
            dtype=torch.long,
            device=device
        )  # shape (t)

        # forward the GPT model itself
        # token embeddings of shape (b, t, n_embd)
        tok_emb = self.transformer.wte(idx)  # type:ignore
        # position embeddings of shape (t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # type:ignore
        x = self.transformer.drop(tok_emb + pos_emb)  # type:ignore
        for block in self.transformer.h:  # type:ignore
            x = block(x)
        x = self.transformer.ln_f(x)  # type:ignore
        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(
                    -1,
                    logits.size(-1)
                ),
                targets.view(-1),
                ignore_index=-1
            )
        else:
            # inference-time mini-optimization: only forward the lm_head on the
            # very last position
            # note: using list [-1] to preserve the time dim
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary e.g. we may
        # load the GPT2 pretrained model checkpoint (block size 1024) but want
        # to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = (  # type:ignore
            nn.Parameter(
                self.transformer.wpe.weight[:block_size]  # type:ignore
            )
        )
        for block in self.transformer.h:   # type:ignore
            if hasattr(block.attn, 'bias'):
                block.attn.bias = (
                    block.attn.bias[:, :, :block_size, :block_size]
                )

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {}  # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        log.info(f"loading weights from pretrained gpt: {model_type=}")
        # n_layer, n_head and n_embd are determined from model_type
        # gpt2: 124M params
        # gpt2-medium: 350M params
        # gpt2-large: 774M params
        # gpt2-xl: 1558M params
        config_args = {
            # 'baby-llama2': dict(n_layer=16, n_head=16, n_embed=1024),
            # 'llama2-7b': dict(n_layer=32, n_head=32, n_embd=4096),
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            log.info(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        log.info("forcing vocab_size=50257, block_size=1024, bias=True")
        config = GPTModelConfig(
            **config_args,
            block_size=1024,   # always 1024 for GPT model checkpoints
            vocab_size=50257,  # always 50257 for GPT model checkpoints
            bias=True,         # always True for GPT model checkpoints
        )
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [
            k for k in sd_keys if not k.endswith('.attn.bias')
        ]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in
        # names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')
        ]  # ignore these, just a buffer
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith('.attn.bias')
        ]  # same, just the mask (buffer)
        transposed = [
            'attn.c_attn.weight',
            'attn.c_proj.weight',
            'mlp.c_fc.weight',
            'mlp.c_proj.weight'
        ]
        # basically the openai checkpoints use a "Conv1D" module, but we only
        # want to use a vanilla Linear this means that we have to transpose
        # these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), (
            f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        )
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(
            self,
            weight_decay,
            learning_rate,
            betas,
            device_type
    ):
        # start with all of the candidate parameters
        # filter out those that do not require grad
        # param_dict = {
        #     pn: p for pn, p in param_dict.items() if p.requires_grad
        # }
        param_dict = {
            pn: p for pn, p in self.named_parameters() if p.requires_grad
        }
        # create optim groups. Any parameters that is 2D will be weight
        # decayed, otherwise no. i.e. all weight tensors in matmuls +
        # embeddings decay, all biases and layernorms don't.
        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        log.info(
            f"num decayed parameter tensors: {len(decay_params)}, "
            f"with {num_decay_params:,} parameters"
        )
        log.info(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, "
            f"with {num_nodecay_params:,} parameters"
        )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = (
            'fused' in inspect.signature(torch.optim.AdamW).parameters
        )
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else {}
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=betas,
            **extra_args
        )
        log.info(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """Estimate model flops utilization (MFU)

        (in units of A100 bfloat16 peak FLOPS)
        """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = (
            cfg.n_layer,
            cfg.n_head,
            cfg.n_embd//cfg.n_head,
            cfg.block_size
        )
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        return flops_achieved / flops_promised

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t))
        and complete the sequence max_new_tokens times, feeding the predictions
        back into the model each time.

        Most likely you'll want to make sure to be in model.eval() mode of
        operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at
            # block_size
            idx_cond = (
                idx if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size:]
            )
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired
            # temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
```

</div>

# Trainer [ `trainer.py`](https://github.com/saforem2/wordplay/blob/master/src/wordplay/trainer.py)

<div class="panel-tabset"
style="font-size: 0.75em; width: 100%; height: 100%;">

### `_forward_step`

```` python
"""
wordplay/trainer.py

```markdown
> [!NOTE]
>  If your cluster does not have Infiniband interconnect, prepend:
>  `NCCL_IB_DISABLE=1`
```
"""
from __future__ import absolute_import, annotations, division, print_function
from dataclasses import asdict
import logging
import math
from os import PathLike
import os
from pathlib import Path
import time
from typing import Any, Optional, Union

from ezpz import (
    get_local_rank,
    get_rank,
    get_torch_device,
    get_world_size,
    timeitlogit
)
from ezpz.history import BaseHistory
import numpy as np
from rich.table import Table
from rich.text import Text
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import trange
import wandb

from wordplay.configs import ExperimentConfig, GPTModelConfig, add_to_ckpts_file
from wordplay.model import GPT


log = logging.getLogger(__name__)

RANK = get_rank()
WORLD_SIZE = get_world_size()
DEVICE = os.environ.get('TORCH_DEVICE', get_torch_device())
# DEVICE = get_torch_device()  # 'cuda' if torch.cuda.is_available() else 'cpu'

ScalarLike = Union[float, int, np.floating, bool]


def print_legend(verbose: bool = True) -> Table:
    legend = {
        "step": "Current training iteration",
        "loss": "Loss value",
        "dt": "Elapsed time per training step (measured in **ms**)",
        "dtf": "Elapsed time per forward step (measured in **ms**)",
        "dtb": "Elapsed time per backward step (measured in **ms**)",
        "sps": "Samples per second",
        "mtps": "Tokens per second, measured in MEGA (1 x 10^6) tokens / sec",
        "mfu": "Model flops utilization",
        "train_loss": "Training loss value",
        "val_loss": "Validation loss value",
    }
    table = Table(title='Training Legend')
    table.add_column('abbr', justify='center', style='green')
    table.add_column('desc', justify='left')
    for key, val in legend.items():
        table.add_row(f'{key}', f'{val}')
    if verbose and RANK == 0:
        from rich import print
        print(table)
    return table


def markdown_legend() -> None:
    from rich.markdown import Markdown
    from rich import print
    text = """
    | name | description |
    | :--: | ---- |
    | `step` | Current training iteration |
    | `loss` | Loss value |
    | `dt` | Elapsed time per training step (measured in **ms**) |
    | `dtf` | Elapsed time per forward step (measured in **ms**) |
    | `dtb` | Elapsed time per backward step (measured in **ms**) |
    | `sps` | Samples per second |
    | `mtps` | Tokens per second, measured in MEGA (1 x 10^6) tokens / sec  |
    | `mfu` | Model flops utilization |
    | `train_loss` | Training loss value |
    | `val_loss` | Validation loss value |
    """
    print(Markdown(text))


def format_pair(k: str, v: ScalarLike) -> str:
    if isinstance(v, (int, bool, np.integer)):
        # return f'{k}={v:<3}'
        return f'{k}={v}'
    # return f'{k}={v:<3.4f}'
    return f'{k}={v:<6.4f}'


def summarize_dict(d: dict) -> str:
    return ' '.join([format_pair(k, v) for k, v in d.items()])


def grab_tensor(x: Any) -> np.ndarray | ScalarLike | None:
    if x is None:
        return None
    if isinstance(x, (int, float, bool, np.floating)):
        return x
    if isinstance(x, list):
        if isinstance(x[0], torch.Tensor):
            return grab_tensor(torch.stack(x))
        elif isinstance(x[0], np.ndarray):
            return np.stack(x)
        else:
            import tensorflow as tf
            if isinstance(x[0], tf.Tensor):
                return grab_tensor(tf.stack(x))
    elif isinstance(x, np.ndarray):
        return x
    elif isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif callable(getattr(x, 'numpy', None)):
        assert callable(getattr(x, 'numpy'))
        return x.numpy()
    raise ValueError


def _average(val):
    if isinstance(val, (list, tuple)):
        if isinstance(val[0], torch.Tensor):
            val = grab_tensor(torch.stack(val))
        elif isinstance(val, np.ndarray):
            val = np.stack(val)
        else:
            val = val
    if isinstance(val, torch.Tensor):
        val = grab_tensor(val)

    if isinstance(val, (float, int, bool, np.floating, np.integer)):
        return val
    try:
        avg = np.mean(val).real  # type: ignore
        assert isinstance(avg, np.floating)
        return avg
    except Exception:
        log.exception(f'Failed to average {val}')
        log.warning('Returning val as is')
        return val


def average_dict(d: dict) -> dict:
    avgs = {}
    avg = 0.0
    for key, val in d.items():
        if val is None:
            continue
        if isinstance(val, dict):
            for k, v in val.items():
                kk = f'{key}/{k}'
                avg = _average(v)
                avgs[kk] = avg
        else:
            avg = _average(val)
            avgs[key] = avg
    return avgs


def GPT_from_pretrained(
        init_from: str,
        dropout: Optional[float] = None,
) -> tuple[GPTModelConfig, GPT]:
    log.info(
        'Initializing from OpenAI GPT-2 Weights: '
        f'{init_from=}'
    )
    override_args = {'dropout': dropout}
    model = GPT.from_pretrained(
        init_from,
        override_args
    )
    model_cfg = {
        k: getattr(model.config, k) for k in [
            'n_layer',
            'n_head',
            'n_embd',
            'block_size',
            'bias',
            'vocab_size'
        ]
    }
    return (model, GPTModelConfig(**model_cfg))


# def setup_deepspeed(
#             model: Optional[torch.nn.Module | GPT],
#             micro_batch_size: Optional[int] = None,
#             ds_config: Optional[dict] = None,
#             ds_config_path: Optional[os.PathLike] = None,
#             optimizer: Optional[torch.optim.Optimizer] = None,
# ) -> dict:
#     import deepspeed
#     from ezpz import load_ds_config
#     if ds_config is None:
#         assert ds_config_path is not None, (
#             'One of `ds_config` or `ds_config_path` must be specified.'
#         )
#         ds_config = load_ds_config(Path(ds_config_path).as_posix())
#     assert ds_config is not None
#     if self.config.train.wandb_project is not None:
#         ds_config['wandb'].update({
#             'enabled': True,
#             'project': self.config.train.wandb_project,
#         })
#     # log.warning(
#     #     f'Setting `train_micro_batch_size_per_gpu` to '
#     #     f'{self.config.model.batch_size=}'
#     # )
#     if micro_batch_size is not None:
#         ds_config.update({
#             'train_micro_batch_size_per_gpu': micro_batch_size
#         })
#     assert (
#         model is not None and (
#             # isinstance(model, (torch.nn.Module, GPT))
#             issubclass(model, torch.nn.Module)
#         )
#     )
#     # assert model is not None
#     if (
#             optimizer is not None
#             and isinstance(optimizer, torch.optim.Optimizer)
#     ):
#         engine, optimizer, *_ = deepspeed.initialize(
#             model=model,
#             config=ds_config,
#             optimizer=optimizer,
#         )
#     elif 'optimizer' in ds_config.keys():
#         engine, optimizer, *_ = deepspeed.initialize(
#             model=model,
#             config=ds_config,
#             model_parameters=model.parameters()
#         )
#     else:
#         raise ValueError('Unable to initialize DeepSpeed')
#     assert engine is not None and optimizer is not None
#     return {
#         'model_engine': engine,
#         'optimizer': optimizer,
#         'ds_config': ds_config,
#     }


class Trainer:
    def __init__(self, config: ExperimentConfig, device: Optional[str] = None):
        # self.console = get_console()
        self.config = config
        self.ckpt = None
        self.rank = RANK
        self.world_size = WORLD_SIZE
        self.device = device if device is not None else DEVICE
        # assert self.device == self.config.device_type
        # NOTE: ---------------------------------------------------------
        # config.optimizer.gas = (
        #     1 if config.optimizer.gradient_accumulation_steps is None
        #     else config.optimizer.gradient_accumulation_steps
        # ) -------------------------------------------------------------
        self.train_history = BaseHistory()
        self._gas = self.config.optimizer.gas
        self._lr = self.config.optimizer.learning_rate
        self._min_lr = self.config.optimizer.min_lr
        self._diters = self.config.optimizer.lr_decay_iters
        self._witers = self.config.train.warmup_iters
        if self.config.train.init_from == 'scratch':
            log.info('Initializing a new model from scratch')
            model = GPT(self.config.model)
        elif self.config.train.init_from == 'resume':
            model, ckpt = self.restore_from_ckpt()
            self.ckpt = ckpt
            self.config.set_iter_num(ckpt.get('iter_num', 1))
            self.config.set_best_val_loss(ckpt.get('best_val_loss', 1e9))
        elif self.config.train.init_from.startswith('gpt2'):
            model = self._init_gpt2()
        else:
            raise ValueError(
                f'Unexpected `init_from` = {self.config.train.init_from}. '
                'Exiting!'
            )
        # model = model
        # if torch.cuda.is_available():
        #     model.cuda()
        model.to(self.device)
        assert isinstance(model, GPT)
        assert issubclass(GPT, torch.nn.Module)
        num_params = model.get_num_params()
        if wandb.run is not None:
            wandb.watch(model)
            wandb.run.config['num_params'] = num_params
        # model_block_size = int(self.model.config.block_size)
        if self.config.model.block_size < model.config.block_size:
            model.crop_block_size(self.config.model.block_size)
            self.config.model.set_block_size(self.config.model.block_size)
        optimizer = model.configure_optimizers(
            weight_decay=self.config.optimizer.weight_decay,
            learning_rate=self.config.optimizer.learning_rate,
            betas=(
                self.config.optimizer.beta1,
                self.config.optimizer.beta2,
            ),
            device_type=self.config.device_type,
        )
        if self.config.train.init_from == 'resume':
            assert (
                self.ckpt is not None
                and isinstance(self.ckpt, dict)
                and 'optimizer' in self.ckpt
            )
            optimizer.load_state_dict(self.ckpt['optimizer'])
            self.ckpt = None  # free up memory
        if self.config.train.compile:
            # unoptimized_model = self.model
            model = torch.compile(model)  # type:ignore
        # if WORLD_SIZE > 1:
        grad_scaler = None
        if self.config.train.backend.lower() == 'ddp':
            if torch.cuda.is_available():
                from torch.cuda.amp.grad_scaler import GradScaler
                grad_scaler = GradScaler(
                    enabled=(self.config.train.dtype == 'float16')
                )
            # self.optimizer = optimizer
            assert isinstance(model, torch.nn.Module)
            # device = get_torch_device()
            local_rank = get_local_rank()
            devid = f"{self.device}:{local_rank}"
            log.critical(f'"{devid=}"')
            model.to(devid)
            if WORLD_SIZE > 1:
                model_engine = DDP(model, device_ids=[devid])
            else:
                model_engine = model
        elif self.config.train.backend.lower() in ['deepspeed', 'ds']:
            from ezpz import load_ds_config
            grad_scaler = None
            ds_config_path = self.config.train.ds_config_path
            if ds_config_path is None:
                from wordplay.configs import DS_CONFIG_PATH
                ds_config_path = DS_CONFIG_PATH
            self.ds_config = load_ds_config(ds_config_path)
            if 'optimizer' in self.ds_config.keys():
                optimizer = None
            assert isinstance(model, torch.nn.Module)
            ds_out = self._setup_deepspeed(
                ds_config=self.ds_config,
                model=model,
                optimizer=optimizer
            )
            model_engine = ds_out['model_engine']
            optimizer = ds_out['optimizer']
        else:
            raise ValueError(f'Unexpected {self.config.train.backend=}')
        self.model = model
        self.grad_scaler = grad_scaler
        self.model_engine = model_engine
        self.optimizer = optimizer

    def _init_gpt2(self) -> GPT:
        log.info(
            f'Initializing from OpenAI GPT-2 Weights: '
            f'{self.config.train.init_from}'
        )
        model_cfg, model = GPT_from_pretrained(
            self.config.train.init_from,
            self.config.model.dropout
        )
        self.config.reset_model_config(model_cfg)
        return model

    def _setup_deepspeed(
            self,
            model: Optional[torch.nn.Module | GPT],
            ds_config: Optional[dict] = None,
            ds_config_path: Optional[os.PathLike] = None,
            optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> dict:
        """Setup DeepSpeed.

        TODO:
            - [ ] Deal with / fix gradient accumulation logic in `train_step`
            - [ ] Test / generalize optimizer creation
        """
        import deepspeed
        from ezpz import load_ds_config
        if ds_config is None:
            assert ds_config_path is not None, (
                'One of `ds_config` or `ds_config_path` must be specified.'
            )
            ds_config = load_ds_config(Path(ds_config_path).as_posix())
        assert ds_config is not None
        if self.config.train.wandb_project is not None:
            ds_config['wandb'].update({
                'enabled': True,
                'project': self.config.train.wandb_project,
            })
        log.warning(
            f'Setting `train_micro_batch_size_per_gpu` to '
            f'{self.config.model.batch_size=}'
        )
        ds_config.update({
            'train_micro_batch_size_per_gpu': self.config.model.batch_size
        })
        ds_config |= {'steps_per_print': self.config.train.log_interval}
        assert (
            model is not None and (
                isinstance(model, (torch.nn.Module, GPT))
                or issubclass(model, torch.nn.Module)
            )
        )
        assert model is not None
        if (
                optimizer is not None
                and isinstance(optimizer, torch.optim.Optimizer)
        ):
            engine, optimizer, *_ = deepspeed.initialize(
                model=model,
                config=ds_config,
                optimizer=optimizer,
            )
        elif 'optimizer' in ds_config.keys():
            engine, optimizer, *_ = deepspeed.initialize(
                model=model,
                config=ds_config,
                model_parameters=model.parameters()
            )
        else:
            raise ValueError('Unable to initialize DeepSpeed')
        assert engine is not None and optimizer is not None
        return {
            'model_engine': engine,
            'optimizer': optimizer,
            'ds_config': ds_config,
        }

    def get_batch(self, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        # data = self.config.train_data if split == 'train'
        # else self.config.val_data
        data = self.config.data.data.get(split, None)
        assert data is not None
        ix = torch.randint(
            len(data) - self.config.model.block_size,
            (self.config.model.batch_size,)
        )
        block_size = self.config.model.block_size
        x = torch.stack(
            [
                torch.from_numpy((data[i:i+block_size]).astype(np.int64))
                for i in ix
            ]
        )
        y = torch.stack(
            [
                torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64))
                for i in ix
            ]
        )
        if self.config.device_type == 'cuda':
            x = x.pin_memory().to(self.config.device_type, non_blocking=True)
            y = y.pin_memory().to(self.config.device_type, non_blocking=True)
        else:
            x = x.to(self.config.device_type)
            y = y.to(self.config.device_type)
        return x, y

    def get_lr(self, it: int) -> float:
        if it < self._witers:
            return self._lr * it / self._witers
        if it > self._diters:
            return self._min_lr
        decay_ratio = (it - self._witers) / (self._diters - self._witers)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self._min_lr + coeff * (self._lr - self._min_lr)

    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        self.model.eval()
        for split in self.config.data.data.keys():
            losses = torch.zeros(self.config.train.eval_iters)
            for k in range(self.config.train.eval_iters):
                x, y = self.get_batch(split)
                with self.config.ctx:
                    _, loss = self.model_engine(x, y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out

    def restore_from_ckpt(
            self,
            ckpt_dir: Optional[str | PathLike] = None
    ) -> tuple[torch.nn.Module, dict]:
        log.info(f'Resuming training from {self.config.data.out_dir}')
        ckpt_dir = (
            str(self.config.data.out_dir) if ckpt_dir is None
            else ckpt_dir
        )
        assert ckpt_dir is not None
        ckpt_path = Path(ckpt_dir).joinpath('ckpt.pt')
        checkpoint = torch.load(
            ckpt_path,
            map_location=self.config.train.device
        )
        ckpt_model = checkpoint['model_args']
        model_config = GPTModelConfig(
            n_layer=ckpt_model['n_layer'],
            n_head=ckpt_model['n_head'],
            n_embd=ckpt_model['n_embd'],
            block_size=ckpt_model['block_size'],
            bias=ckpt_model['bias'],
            vocab_size=ckpt_model['vocab_size'],
        )
        model = GPT(model_config)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k, _ in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        return model, checkpoint

    def _forward_step(self, x: torch.Tensor, y: torch.Tensor) -> dict:
        t0 = time.perf_counter()
        with self.config.ctx:
            logits, loss = self.model_engine(x, y)
        return {
            'logits': logits,
            'loss': loss,
            'dt': time.perf_counter() - t0
        }

    def _backward_step(
            self,
            loss: torch.Tensor,
            propagate_grads: bool = False,
    ) -> float:
        t0 = time.perf_counter()
        if self.config.train.backend.lower() in ['ds', 'deepspeed']:
            self.model_engine.backward(loss)  # type:ignore
            self.model_engine.step(loss)      # type:ignore
        else:
            if self.grad_scaler is not None:
                self.grad_scaler.scale(loss).backward()  # type:ignore
            if propagate_grads:
                if self.config.optimizer.grad_clip != 0.0:
                    if self.grad_scaler is not None:
                        self.grad_scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(  # pyright: ignore
                        self.model_engine.parameters(),
                        self.config.optimizer.grad_clip
                    )
                if self.grad_scaler is not None:
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)

        return time.perf_counter() - t0

    def train_step(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
    ) -> dict:
        lr = (
            self.get_lr(self.config.iter_num)
            if self.config.optimizer.decay_lr
            else self._lr
        )
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        dtf = []
        dtb = []
        dt = []
        loss = torch.tensor(0.0)
        for micro_step in range(self._gas):
            is_last_micro_step = (micro_step == self._gas - 1)
            # NOTE: -----------------------------------------------------------
            # In DDP training we only need to sync gradients at the last micro
            # step. the official way to do this is with model.no_sync() context
            # manager, but I really dislike that this bloats the code and
            # forces us to repeat code looking at the source of that context
            # manager, it just toggles this variable
            # -----------------------------------------------------------------
            if self.config.train.backend.lower() == 'ddp':
                _ = (
                    self.model_engine.require_backward_grad_sync
                    if (is_last_micro_step and self.world_size > 1)
                    else None
                )
            fout = self._forward_step(x, y)
            # immediately async prefetch next batch while model is doing the
            # forward pass on the GPU
            x, y = self.get_batch('train')
            loss = fout['loss'] / self._gas
            dtf.append(fout['dt'])
            dtb_ = self._backward_step(
                loss,
                propagate_grads=is_last_micro_step
            )
            dtb.append(dtb_)
            dt.append(dtf + dtb)
        timers = {
            'iter': self.config.iter_num,
            'dt': np.array(dt),
            'dt_tot': np.sum(dt),
            'dt_avg': np.mean(dt),
            'dtf': np.array(dtf),
            'dtf_tot': np.sum(dtf),
            'dtf_avg': np.mean(dtf),
            'dtb': np.array(dtb),
            'dtb_tot': np.sum(dtb),
            'dtb_avg': np.mean(dtb)
        }
        metrics = {
            'iter': self.config.iter_num,
            'loss': loss,
            'lr': lr,
        }
        self.config.iter_num += 1
        return {
            'metrics': metrics,
            'timers': timers,
            'x': x,
            'y': y,
        }

    def save_ckpt(
            self,
            raw_model: Optional[torch.nn.Module | GPT] = None,
            add_to_wandb: bool = False
    ):
        if raw_model is not None:
            model = raw_model  # type:ignore
        else:
            model = self.model
        assert model is not None
        assert isinstance(model, torch.nn.Module)
        # assert issubclass(GPT,  torch.nn.Module)
        ckpt = {
            'model': model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'model_args': asdict(self.config.model),
            'iter_num': self.config.iter_num,
            'best_val_loss': self.config.best_val_loss,
            'config': asdict(self.config),
        }
        # assert (
        #     isinstance(model, GPT)
        #     and issubclass(GPT, torch.nn.Module)
        # )
        # assert raw_model is not None
        ckptfile = Path(os.getcwd()).joinpath('ckpt.pt')
        modelfile = Path(os.getcwd()).joinpath('model.pth')
        log.info(f'Saving checkpoint to: {os.getcwd()}')
        log.info(f'Saving model to: {modelfile}')
        torch.save(model.state_dict(), modelfile.as_posix())
        torch.save(ckpt, ckptfile.as_posix())
        add_to_ckpts_file(Path(os.getcwd()))
        if add_to_wandb and wandb.run is not None:
            artifact = wandb.Artifact('model', type='model')
            artifact.add_file(modelfile.as_posix())
            wandb.run.log_artifact(artifact)

    @timeitlogit(rank=RANK, verbose=(RANK != 0))
    def train(
            self,
            train_iters: Optional[int] = None,
    ):
        x, y = self.get_batch('train')
        t0 = time.perf_counter()
        running_mfu = -1.0
        output = {'x': x, 'y': y}
        t0 = time.perf_counter()
        losses = {}
        train_iters = (
            self.config.train.max_iters
            if train_iters is None else train_iters
        )
        for train_iter in trange(
                train_iters,
                disable=(self.rank != 0),
                total=train_iters,
        ):
            if self.config.iter_num == 0:
                start_time = os.environ.get('START_TIME', None)
                if start_time is not None:
                    startup_time = time.perf_counter() - float(start_time)
                    log.info(f'Startup time: {startup_time:.4f}')
                    if wandb is not None and wandb.run is not None:
                        wandb.run.log(
                            {'Timing/startup_time': startup_time},
                            commit=False
                        )
                _ = print_legend()
                # markdown_legend()
            if self.config.iter_num == 0 and self.config.train.eval_only:
                return
            if (
                    self.config.iter_num % self.config.train.eval_interval == 0
                    and self.rank == 0
            ):
                losses = self.estimate_loss()
                if (
                    self.config.iter_num > 0
                    and (losses.get('val', np.inf) < self.config.best_val_loss
                         or self.config.train.always_save_checkpoint)
                ):
                    self.save_ckpt(add_to_wandb=False)
            output = self.train_step(x=output['x'], y=output['y'])
            t1 = time.perf_counter()
            dt = t1 - t0
            tokens_per_sec = self.config.tokens_per_iter / dt
            samples_per_sec = self.config.samples_per_iter / dt
            t0 = t1
            output['timers'] |= {
                'dt_iter': dt,
                'tokens_per_sec': tokens_per_sec,
                'samples_per_sec': samples_per_sec,
            }
            # metrics = output['metrics']
            # metrics |= output['timers']
            lossf = output['metrics']['loss'].item() * self._gas
            output['metrics']['loss_tot'] = lossf
            _ = self.train_history.update(output['timers'])
            _ = self.train_history.update(output['metrics'])
            zero = torch.tensor(0.0)
            if (
                    self.config.iter_num % self.config.train.log_interval == 0
                    and (self.rank == 0)
            ):
                if train_iter >= 5:
                    mfu = self.model.estimate_mfu(
                        (
                            self.config.model.batch_size
                            * self.config.optimizer.gas
                        ),
                        dt=dt
                    )
                    running_mfu = (
                        mfu if running_mfu == -1.0
                        else 0.9 * running_mfu + 0.1 * mfu
                    )
                pvars = {
                    'step': self.config.iter_num,
                    'loss': lossf,
                    'dt': dt * 1000,
                    'dtf': output['timers']['dtf_avg'] * 1000,
                    'dtb': output['timers']['dtb_avg'] * 1000,
                    'sps': samples_per_sec,
                    'mtps': tokens_per_sec / int(1e6),
                    'mfu': running_mfu * 100,
                    'train_loss': losses.get('train', zero).item(),
                    'val_loss': losses.get('val', zero).item(),
                }
                summary = summarize_dict(pvars)
                log.info(Text(summary))
                if wandb.run is not None:
                    losses |= {
                        'lossf': lossf,
                        'mfu': running_mfu * 100,
                        'iter': self.config.iter_num,
                    }
                    losses['lossf'] = lossf
                    losses['iter'] = self.config.iter_num
                    wbmetrics = {
                        f'Training/{k}': (
                            (wandb.Histogram(v.tolist())
                                if isinstance(v, np.ndarray) else v)
                        ) for k, v in output['metrics'].items()
                    }
                    wbmetrics |= {
                        f'Timing/{k}': (
                            (wandb.Histogram(v.tolist())
                                if isinstance(v, np.ndarray) else v)
                        ) for k, v in output['timers'].items()
                    }
                    wbmetrics |= {
                        f'Loss/{k}': v for k, v in losses.items()
                    }
                    wandb.run.log(wbmetrics)
                    # wandb.run.log({
                    #     'losses': losses,
                    #     'metrics': output['metrics'],
                    #     'timers': output['timers'],
                    #     # 'training': metrics,
                    # })

    def unwrap_model_engine(self) -> torch.nn.Module:
        if hasattr(self.model, 'module'):
            return self.model.module
        else:
            return self.model

    def evaluate(
            self,
            s: str,
            num_samples: int = 10,
            max_new_tokens: int = 500,
            temperature: float = 0.8,
            top_k: int = 200,
            display: Optional[bool] = True,
    ) -> dict[str, str]:
        # seed: Optional[int] = None,
        # assert isinstance(self.model.module, GPT)
        # assert issubclass(GPT, torch.nn.Module)
        model = self.unwrap_model_engine()
        model.eval()
        outputs = {}
        with torch.no_grad():
            start_ids = self.config.data.encode(s)
            x = torch.tensor(
                    start_ids,
                    dtype=torch.long,
                    device=self.device,
            )[None, ...]
            for idx in range(num_samples):
                # y = self.model.module.generate(
                y = model.generate(
                    x,
                    max_new_tokens,
                    temperature=temperature,
                    top_k=top_k
                )
                response = self.config.data.decode(y[0].tolist())
                # outputs.append(response)
                response_ = [i for i in response.split('\n')]
                prompt = response_[0]
                responses = [*response_[1:]]
                ret0 = fr"[prompt]: '{prompt}'"
                ret1 = '> ' + '\n> '.join(responses)
                if display:
                    log.info(f'{ret0}')
                    log.info(f'{ret1}')
                outputs[f'{idx}'] = {
                    'raw': response,
                    'prompt': Text(ret0, style='string'),
                    'formatted': Text(ret1, style='blockquote'),
                }
                # log.info(f'[prompt]: "{s}"')
                # # responses = reponse.split('\n ')
                # log.info('> "' + '\n> '.join(response.split('\n ')) + '"')
                #
                # log.info('\n'.join)
                # log.info(f'> "{response}"')
                # log.info(100 * '-')
        return outputs
````

### `_backward_step`

``` python
    def _backward_step(
            self,
            loss: torch.Tensor,
            propagate_grads: bool = False,
    ) -> float:
        t0 = time.perf_counter()
        if self.config.train.backend.lower() in ['ds', 'deepspeed']:
            self.model_engine.backward(loss)  # type:ignore
            self.model_engine.step(loss)      # type:ignore
        else:
            if self.grad_scaler is not None:
                self.grad_scaler.scale(loss).backward()  # type:ignore
            if propagate_grads:
                if self.config.optimizer.grad_clip != 0.0:
                    if self.grad_scaler is not None:
                        self.grad_scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(  # pyright: ignore
                        self.model_engine.parameters(),
                        self.config.optimizer.grad_clip
                    )
                if self.grad_scaler is not None:
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)

        return time.perf_counter() - t0
```

### `train_step`

``` python
    def train_step(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
    ) -> dict:
        lr = (
            self.get_lr(self.config.iter_num)
            if self.config.optimizer.decay_lr
            else self._lr
        )
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        dtf = []
        dtb = []
        dt = []
        loss = torch.tensor(0.0)
        for micro_step in range(self._gas):
            is_last_micro_step = (micro_step == self._gas - 1)
            # NOTE: -----------------------------------------------------------
            # In DDP training we only need to sync gradients at the last micro
            # step. the official way to do this is with model.no_sync() context
            # manager, but I really dislike that this bloats the code and
            # forces us to repeat code looking at the source of that context
            # manager, it just toggles this variable
            # -----------------------------------------------------------------
            if self.config.train.backend.lower() == 'ddp':
                _ = (
                    self.model_engine.require_backward_grad_sync
                    if (is_last_micro_step and self.world_size > 1)
                    else None
                )
            fout = self._forward_step(x, y)
            # immediately async prefetch next batch while model is doing the
            # forward pass on the GPU
            x, y = self.get_batch('train')
            loss = fout['loss'] / self._gas
            dtf.append(fout['dt'])
            dtb_ = self._backward_step(
                loss,
                propagate_grads=is_last_micro_step
            )
            dtb.append(dtb_)
            dt.append(dtf + dtb)
        timers = {
            'iter': self.config.iter_num,
            'dt': np.array(dt),
            'dt_tot': np.sum(dt),
            'dt_avg': np.mean(dt),
            'dtf': np.array(dtf),
            'dtf_tot': np.sum(dtf),
            'dtf_avg': np.mean(dtf),
            'dtb': np.array(dtb),
            'dtb_tot': np.sum(dtb),
            'dtb_avg': np.mean(dtb)
        }
        metrics = {
            'iter': self.config.iter_num,
            'loss': loss,
            'lr': lr,
        }
        self.config.iter_num += 1
        return {
            'metrics': metrics,
            'timers': timers,
            'x': x,
            'y': y,
        }

    def save_ckpt(
            self,
            raw_model: Optional[torch.nn.Module | GPT] = None,
            add_to_wandb: bool = False
    ):
        if raw_model is not None:
            model = raw_model  # type:ignore
        else:
            model = self.model
        assert model is not None
        assert isinstance(model, torch.nn.Module)
        # assert issubclass(GPT,  torch.nn.Module)
        ckpt = {
            'model': model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'model_args': asdict(self.config.model),
            'iter_num': self.config.iter_num,
            'best_val_loss': self.config.best_val_loss,
            'config': asdict(self.config),
        }
        # assert (
        #     isinstance(model, GPT)
        #     and issubclass(GPT, torch.nn.Module)
        # )
        # assert raw_model is not None
        ckptfile = Path(os.getcwd()).joinpath('ckpt.pt')
        modelfile = Path(os.getcwd()).joinpath('model.pth')
        log.info(f'Saving checkpoint to: {os.getcwd()}')
        log.info(f'Saving model to: {modelfile}')
        torch.save(model.state_dict(), modelfile.as_posix())
        torch.save(ckpt, ckptfile.as_posix())
        add_to_ckpts_file(Path(os.getcwd()))
        if add_to_wandb and wandb.run is not None:
            artifact = wandb.Artifact('model', type='model')
            artifact.add_file(modelfile.as_posix())
            wandb.run.log_artifact(artifact)
```

### `get_batch`

``` python
    def get_batch(self, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        # data = self.config.train_data if split == 'train'
        # else self.config.val_data
        data = self.config.data.data.get(split, None)
        assert data is not None
        ix = torch.randint(
            len(data) - self.config.model.block_size,
            (self.config.model.batch_size,)
        )
        block_size = self.config.model.block_size
        x = torch.stack(
            [
                torch.from_numpy((data[i:i+block_size]).astype(np.int64))
                for i in ix
            ]
        )
        y = torch.stack(
            [
                torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64))
                for i in ix
            ]
        )
        if self.config.device_type == 'cuda':
            x = x.pin_memory().to(self.config.device_type, non_blocking=True)
            y = y.pin_memory().to(self.config.device_type, non_blocking=True)
        else:
            x = x.to(self.config.device_type)
            y = y.to(self.config.device_type)
        return x, y
```

### `estimate_loss`

``` python
    def estimate_loss(self):
        out = {}
        self.model.eval()
        for split in self.config.data.data.keys():
            losses = torch.zeros(self.config.train.eval_iters)
            for k in range(self.config.train.eval_iters):
                x, y = self.get_batch(split)
                with self.config.ctx:
                    _, loss = self.model_engine(x, y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out

```

</div>

# Hands-on Tutorial

<div>

</div>

# 

# Links

1.  [
    Hannibal046/Awesome-LLM](https://github.com/Hannibal046/Awesome-LLM/blob/main/README.md)
    <span class="inline-image">[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)</span>
2.  [
    Mooler0410/LLMsPracticalGuide](https://github.com/Mooler0410/LLMsPracticalGuide)
3.  [Large Language Models (in
    2023)](https://docs.google.com/presentation/d/1636wKStYdT_yRPbJNrf8MLKpQghuWGDmyHinHhAKeXY/edit#slide=id.g238b2698243_0_734https://docs.google.com/presentation/d/1636wKStYdT_yRPbJNrf8MLKpQghuWGDmyHinHhAKeXY/edit#slide=id.g238b2698243_0_734)
4.  [The Illustrated
    Transformer](http://jalammar.github.io/illustrated-transformer/)
5.  [Generative AI Exists because of the
    Transformer](https://ig.ft.com/generative-ai/)
6.  [GPT in 60 Lines of
    Numpy](https://jaykmody.com/blog/gpt-from-scratch/)
7.  [Better Language Models and their
    Implications](https://openai.com/research/better-language-models)  
8.  <span class="green-text"></span> [Progress / Artefacts / Outcomes
    from 🌸 Bloom
    BigScience](https://bigscience.notion.site/ebe3760ae1724dcc92f2e6877de0938f?v=2faf85dc00794321be14bc892539dd4f)

> [!NOTE]
>
> ### Acknowledgements
>
> This research used resources of the Argonne Leadership Computing
> Facility,  
> which is a DOE Office of Science User Facility supported under
> Contract DE-AC02-06CH11357.

# References

<div id="refs" class="references csl-bib-body hanging-indent"
entry-spacing="0">

<div id="ref-yao2023tree" class="csl-entry">

Yao, Shunyu, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths,
Yuan Cao, and Karthik Narasimhan. 2023. “Tree of Thoughts: Deliberate
Problem Solving with Large Language Models.”
<https://arxiv.org/abs/2305.10601>.

</div>

</div>

[^1]: [`saforem2/LLM-tutorial`](https://github.com/saforem2/LLM-tutorial)
