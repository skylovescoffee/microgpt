# Model

Architecture and data flow of the MicroGPT transformer.

## Overview

MicroGPT is a minimal GPT-2–style decoder-only transformer. It trains and runs in pure Python with no external dependencies. The implementation uses a custom autograd engine (`Value` class) for backpropagation. See `checkpoint.md` for parameter layout and math details.

## Architecture

```
Input tokens → Embedding (token + position) → RMSNorm
    → [Transformer block × n_layer]
        → Attention (pre-norm, residual)
        → MLP (pre-norm, residual)
    → lm_head → logits → softmax → next-token probs
```

Default config: `n_embd=16`, `n_head=4`, `n_layer=1`, `block_size=16`, `head_dim=4`.

## Data flow

### Tokenization

Character-level: each character maps to a token ID via `uchars`. BOS (Beginning of Sequence) is prepended and appended to each document. Vocabulary size = `len(uchars) + 1`.

### Forward pass (one position)

For position `pos_id` with token `token_id`:

1. **Embed:** \(x = \text{RMSNorm}(\mathbf{e}_{tok} + \mathbf{e}_{pos})\)
2. **Per layer:** pre-norm → Q/K/V projections → multi-head attention → output projection → residual → pre-norm → MLP → residual
3. **Output:** \(\text{logits} = \text{lm\_head} \cdot x\)

Causality: at position \(t\), attention only sees keys/values from positions \(0 \ldots t\) (via the `keys`/`values` cache). See `gpt()` in `microgpt.py`.

### Multi-head attention

Each head operates on a slice of Q, K, V:

\[
\text{attn\_logits}_t = \frac{\mathbf{q}_h \cdot \mathbf{k}_{h,t}}{\sqrt{d_k}}, \quad
\text{attn} = \text{softmax}(\text{attn\_logits}), \quad
\text{head\_out} = \sum_t \text{attn}_t \cdot \mathbf{v}_{h,t}
\]

Heads are concatenated and projected by `attn_wo`, then added to the residual.

## Training

- **Objective:** next-token prediction. Per document, predict the next token at each position.
- **Loss:** mean cross-entropy over positions: \(\mathcal{L} = -\frac{1}{n}\sum_i \log p(\text{target}_i)\).
- **Optimizer:** Adam (\(\beta_1=0.85\), \(\beta_2=0.99\), \(\epsilon=10^{-8}\)) with linear LR decay from 0.01 to 0.
- **Init:** Gaussian \(\mathcal{N}(0, 0.08)\).

One document per step; sequence length capped by `block_size`.

## Inference

Autoregressive sampling:

1. Start with BOS.
2. For each position: run forward pass → softmax over logits → sample next token (with temperature scaling).
3. Stop on BOS or when `block_size` is reached.

Temperature \(T\): logits are divided by \(T\) before softmax. Lower \(T\) → sharper distribution; higher \(T\) → more random.

## Autograd

The `Value` class implements a scalar computation graph:

- Each `Value` holds `data`, `grad`, and references to children.
- Operations (`+`, `*`, `exp`, `log`, `relu`) record local gradients for the chain rule.
- `backward()` performs a topological sort and propagates gradients from the loss to all parameters.

No PyTorch/JAX; gradients are computed manually via the chain rule.

## Design choices

| Choice | Rationale |
|--------|------------|
| Character-level | Simplest tokenizer; no BPE/subword logic. |
| RMSNorm | GPT-2 style; no bias/affine, simpler than LayerNorm. |
| No biases | Matches GPT-2; reduces parameters. |
| ReLU in MLP | Standard; GELU could be added for closer GPT-2 match. |
| Pre-norm | More stable training than post-norm. |
| Single layer | Minimal; easy to extend via `n_layer`. |

## Index

| Term | Meaning |
|------|---------|
| **Autoregressive** | Generate one token at a time; each step conditions on all previous tokens. |
| **Causal** | Attention mask ensures position \(t\) cannot attend to positions \(> t\). |
| **Decoder-only** | No encoder; predicts next token from prior context (GPT style). |
| **Pre-norm** | Normalization applied before the sublayer, not after; residual path is unscaled. |
| **Temperature** | Scaling factor on logits before sampling; controls randomness. |
