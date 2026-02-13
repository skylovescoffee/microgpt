# Checkpoint

Describes checkpoint layout and underlying model math.

## Layout

JSON payload with two top-level keys:

| Key | Type | Purpose |
|-----|------|---------|
| `uchars` | `list[str]` | Unique characters in the dataset, sorted. Token IDs 0..n-1 map to these. |
| `state_dict` | `dict` | All trainable parameters (floats). |

**Vocabulary:** `vocab_size = len(uchars) + 1`. The extra token is BOS (Beginning of Sequence). BOS ID = `len(uchars)`.

## State dict keys

Default config: `n_embd=16`, `n_head=4`, `n_layer=1`, `block_size=16`, `head_dim=4`.

| Key | Shape | Role |
|-----|-------|------|
| `wte` | `(vocab_size × n_embd)` | Token embeddings |
| `wpe` | `(block_size × n_embd)` | Position embeddings |
| `lm_head` | `(vocab_size × n_embd)` | Unembedding / prediction head |
| `layer{i}.attn_wq` | `(n_embd × n_embd)` | Q projection for attention |
| `layer{i}.attn_wk` | `(n_embd × n_embd)` | K projection |
| `layer{i}.attn_wv` | `(n_embd × n_embd)` | V projection |
| `layer{i}.attn_wo` | `(n_embd × n_embd)` | Output projection |
| `layer{i}.mlp_fc1` | `(4×n_embd × n_embd)` | MLP first layer |
| `layer{i}.mlp_fc2` | `(n_embd × 4×n_embd)` | MLP second layer |

## Forward pass math

### Embedding

\[
x = \text{RMSNorm}(\mathbf{e}_{tok} + \mathbf{e}_{pos})
\]

where \(\mathbf{e}_{tok} = \text{wte}[token\_id]\), \(\mathbf{e}_{pos} = \text{wpe}[pos\_id]\).

### RMSNorm

\[
\text{RMSNorm}(x)_i = \frac{x_i}{\sqrt{\text{mean}(x^2) + \epsilon}}
\]

with \(\epsilon = 10^{-5}\). Replaces LayerNorm (no bias, no affine) per GPT-2 style.

### Scaled dot-product attention

Per head \(h\):

\[
\text{attn\_logits}_t = \frac{\mathbf{q}_h \cdot \mathbf{k}_{h,t}}{\sqrt{d_k}}
\]

\[
\text{attn\_weights} = \text{softmax}(\text{attn\_logits})
\]

\[
\text{head\_out}_j = \sum_t \text{attn\_weights}_t \cdot v_{h,t,j}
\]

Heads are concatenated and projected: `x_attn = linear(concat(heads), attn_wo)`. Residual: `x = x_attn + x_residual`.

### MLP

\[
x = \text{ReLU}(W_1 x) W_2 + x_{residual}
\]

where \(W_1\), \(W_2\) are `mlp_fc1` and `mlp_fc2`. Hidden size is 4×n_embd (GPT-2 style).

### Output

\[
\text{logits} = \text{lm\_head} \cdot x
\]

\[
\text{probs} = \text{softmax}(\text{logits})
\]

## Training

- **Loss:** mean cross-entropy over positions: \(\mathcal{L} = -\frac{1}{n}\sum_i \log p(target_i)\).
- **Optimizer:** Adam with \(\beta_1=0.85\), \(\beta_2=0.99\), \(\epsilon=10^{-8}\).
- **LR schedule:** linear decay from 0.01 to 0 over steps.
- **Init:** Gaussian \(\mathcal{N}(0, 0.08)\).

## Persistence

`load_checkpoint()` and `save_checkpoint()` read/write JSON. `state_dict` values are plain floats; `Value` objects are reconstructed on load for autograd.
