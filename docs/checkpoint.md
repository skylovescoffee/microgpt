# Checkpoint

Describes checkpoint layout and underlying model math.

## Core concepts (newbie-friendly)

### Projections

A **projection** is a linear transformation: you multiply a vector by a weight matrix to get a new vector. Example: `y = W @ x`. The matrix \(W\) "projects" \(x\) into a different space. In transformers, projections reshape the same input into Query, Key, and Value forms so attention can compare and combine them. All projections here are linear (no bias).

### Why the "W" prefix (attn_wq, attn_wk, …)

**W** stands for **weight matrix**. This follows standard ML notation (e.g. Vaswani et al., "Attention is All You Need", 2017): \(W^Q\), \(W^K\), \(W^V\), \(W^O\). So `attn_wq` = "attention weight matrix for Query", `attn_wk` for Key, etc. The `w` is not a separate concept—it's just the conventional name for these learned matrices.

### MLP (Multi-Layer Perceptron)

An **MLP** is a small feedforward network: input → linear → ReLU → linear → output. Each transformer block has one after attention. It adds non-linearity and capacity; the 4× expansion (hidden size = 4×n_embd) is GPT-2 style. Think of it as "process the attended representation before passing it on."

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
| `layer{i}.attn_wq` | `(n_embd × n_embd)` | Q projection; weight matrix for Query |
| `layer{i}.attn_wk` | `(n_embd × n_embd)` | K projection; weight matrix for Key |
| `layer{i}.attn_wv` | `(n_embd × n_embd)` | V projection; weight matrix for Value |
| `layer{i}.attn_wo` | `(n_embd × n_embd)` | Output projection; combines heads back |
| `layer{i}.mlp_fc1` | `(4×n_embd × n_embd)` | MLP first linear layer (expand) |
| `layer{i}.mlp_fc2` | `(n_embd × 4×n_embd)` | MLP second linear layer (compress) |

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

Per token, the input \(x\) is projected into Q, K, V via \(W^Q\), \(W^K\), \(W^V\) (the `attn_wq`, `attn_wk`, `attn_wv` matrices). Per head \(h\):

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

where \(W_1\), \(W_2\) are `mlp_fc1` and `mlp_fc2`. Hidden size is 4×n_embd (GPT-2 style). The MLP expands to 4×n_embd, applies ReLU, then projects back to n_embd.

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

---

## Index

| Term | Meaning |
|------|---------|
| **Adam** | Optimizer that adapts learning rate per parameter using moving averages of gradients; commonly used in transformer training (Kingma & Ba, 2015). |
| **Attention** | Mechanism that lets each token attend to others; computes weighted sums of values based on query–key similarity. |
| **Autograd** | Automatic differentiation—computes gradients for backpropagation without manual derivatives. |
| **BOS** | Beginning of Sequence; special token prepended to inputs to mark the start. |
| **block_size** | Maximum context length (tokens the model can attend to). |
| **Cross-entropy** | Loss that measures how well predicted probabilities match targets; \(-\log p(\text{target})\). |
| **Embedding** | Dense vector representation of a discrete token or position; learned lookup. |
| **head_dim** | Dimension of each attention head; typically `n_embd / n_head`. |
| **Key (K)** | Projection of input used to compare against queries; relevance scoring. |
| **LayerNorm** | Normalizes activations per layer; RMSNorm is a simplified variant (no bias, no affine). |
| **lm_head** | Final linear layer mapping hidden states to vocabulary logits (unembedding). |
| **Logits** | Raw model outputs before softmax; higher values ⇒ higher probability. |
| **MLP** | Multi-layer perceptron; a small feedforward net (linear → ReLU → linear) inside each transformer block. Adds non-linearity and model capacity; the "FFN" or "feedforward" sublayer. |
| **Output (O)** | Final attention projection; combines concatenated head outputs back into a single vector (via `attn_wo`). |
| **Projection** | Linear transformation (matrix multiply); maps a vector to another space. Q, K, V, O are all projections of the same input. |
| **n_embd** | Embedding dimension; size of hidden vectors throughout the model. |
| **n_head** | Number of parallel attention heads in multi-head attention. |
| **n_layer** | Number of transformer blocks stacked. |
| **Query (Q)** | Projection of input used to compute attention scores against keys; "what am I looking for?" |
| **ReLU** | Rectified Linear Unit; \(\max(0, x)\); injects non-linearity. |
| **Residual** | Skip connection; adds block input to output so gradients flow directly. |
| **RMSNorm** | Root Mean Square normalization; scales by \(1/\sqrt{\text{mean}(x^2)}\). |
| **Softmax** | Converts logits to a probability distribution; exponentiates and normalizes. |
| **state_dict** | Dictionary of all trainable parameters (weights, biases). |
| **Token** | Discrete unit of text; here a single character. |
| **Token ID** | Integer index into the vocabulary for a token. |
| **Value (V)** | Projection of input combined via attention weights to produce output; "what do I contribute when selected?" |
| **Vocabulary** | Set of all tokens the model can represent; size = token count + 1 (BOS). |
| **Weight (W)** | Conventional ML notation for a weight matrix; e.g. `attn_wq` = W for Query projection. |
| **wpe** | Position embedding; encodes token position in the sequence. |
| **wte** | Token (word) embedding; maps token IDs to vectors. |
