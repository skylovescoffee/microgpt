"""
The most atomic way to train and inference a GPT in pure, dependency-free Python.
This file is the complete algorithm.
Everything else is just efficiency.

@karpathy
"""

import json
import math
import os
import random

random.seed(42)

# -----------------------------------------------------------------------------
# Checkpoint I/O
# -----------------------------------------------------------------------------

CHECKPOINT_DIR = os.getenv('MICROGPT_CHECKPOINT_DIR', 'checkpoints')
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, 'checkpoint.json')


def _state_dict_to_floats(sd):
    """Convert state dict of Value objects to plain floats for JSON serialization."""
    result = {}
    for k, mat in sd.items():
        result[k] = [[p.data for p in row] for row in mat]
    return result


def _floats_to_state_dict(sd_floats):
    """Convert plain floats back to Value objects for computation."""
    result = {}
    for k, mat in sd_floats.items():
        result[k] = [[Value(v) for v in row] for row in mat]
    return result

def load_checkpoint():
    """Load vocab and model weights from disk, or None if no checkpoint exists."""
    if not os.path.exists(CHECKPOINT_PATH):
        return None
    with open(CHECKPOINT_PATH) as f:
        data = json.load(f)
    return data['uchars'], data['state_dict']


def save_checkpoint(uchars, state_dict):
    """Persist vocab and model weights to disk."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    with open(CHECKPOINT_PATH, 'w') as f:
        json.dump({'uchars': uchars, 'state_dict': _state_dict_to_floats(state_dict)}, f)


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

if not os.path.exists('input.txt'):
    import urllib.request
    urllib.request.urlretrieve(
        'https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt',
        'input.txt',
    )

with open('input.txt') as f:
    raw_lines = f.read().strip().split('\n')
docs = [line.strip() for line in raw_lines if line.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")


# -----------------------------------------------------------------------------
# Tokenizer / Vocab
# -----------------------------------------------------------------------------

checkpoint = load_checkpoint()
if checkpoint:
    uchars, state_dict_floats = checkpoint
    state_dict = _floats_to_state_dict(state_dict_floats)
    BOS = len(uchars)
    vocab_size = len(uchars) + 1
    params = []
    for mat in state_dict.values():
        for row in mat:
            for p in row:
                params.append(p)
    print(f"loaded checkpoint from {CHECKPOINT_PATH}")
else:
    # Build vocab from unique chars in dataset. Each char maps to token id 0..n-1.
    uchars = sorted(set(''.join(docs)))
    BOS = len(uchars)  # special Beginning-of-Sequence token
    vocab_size = len(uchars) + 1  # +1 for BOS

print(f"vocab size: {vocab_size}")


# -----------------------------------------------------------------------------
# Autograd: scalar-valued computation graph with automatic differentiation
# Each Value tracks data, gradient, and children for backprop via chain rule.
# -----------------------------------------------------------------------------

class Value:
    """
    Scalar node in a computation graph. Tracks value (data) and gradient for autodiff.
    """

    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0.0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        # d(a+b)/da = 1, d(a+b)/db = 1
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        # d(a*b)/da = b, d(a*b)/db = a
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other):
        # d/dx (x^n) = n * x^(n-1)
        local_grad = other * self.data ** (other - 1)
        return Value(self.data ** other, (self,), (local_grad,))

    def log(self):
        # d/dx log(x) = 1/x
        return Value(math.log(self.data), (self,), (1 / self.data,))

    def exp(self):
        # d/dx exp(x) = exp(x)
        out = math.exp(self.data)
        return Value(out, (self,), (out,))

    def relu(self):
        # ReLU(x) = max(0,x). Gradient is 1 if x>0 else 0.
        local_grad = 1.0 if self.data > 0 else 0.0
        return Value(max(0, self.data), (self,), (local_grad,))

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other ** -1

    def __rtruediv__(self, other):
        return other * self ** -1

    def backward(self):
        """Backprop: compute gradients for all nodes in the graph via chain rule."""
        # Topological sort: visit children before parents so we can propagate
        # gradients from loss (root) down to leaves (parameters).
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1.0  # d(loss)/d(loss) = 1

        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                # Chain rule: d(loss)/d(child) += d(v)/d(child) * d(loss)/d(v)
                child.grad += local_grad * v.grad


# -----------------------------------------------------------------------------
# Model architecture (GPT-2 style: RMSNorm, no biases, ReLU in MLP)
# -----------------------------------------------------------------------------

n_embd = 16      # embedding dimension
n_head = 4       # number of attention heads
n_layer = 1      # number of transformer layers
block_size = 16  # max sequence length (context window)
head_dim = n_embd // n_head  # dimension per attention head (must divide evenly)


def _init_matrix(nout, nin, std=0.08):
    """Create a 2D weight matrix of Values, initialized with small Gaussians."""
    return [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]


if not checkpoint:
    # Embeddings: token -> vector, position -> vector, final projection to vocab
    state_dict = {
        'wte': _init_matrix(vocab_size, n_embd),   # token embeddings
        'wpe': _init_matrix(block_size, n_embd),   # position embeddings
        'lm_head': _init_matrix(vocab_size, n_embd),  # output projection
    }
    for i in range(n_layer):
        # Attention: Q, K, V projections and output projection
        state_dict[f'layer{i}.attn_wq'] = _init_matrix(n_embd, n_embd)
        state_dict[f'layer{i}.attn_wk'] = _init_matrix(n_embd, n_embd)
        state_dict[f'layer{i}.attn_wv'] = _init_matrix(n_embd, n_embd)
        state_dict[f'layer{i}.attn_wo'] = _init_matrix(n_embd, n_embd)
        # MLP: 4x expansion (GPT-2 standard), then project back
        state_dict[f'layer{i}.mlp_fc1'] = _init_matrix(4 * n_embd, n_embd)
        state_dict[f'layer{i}.mlp_fc2'] = _init_matrix(n_embd, 4 * n_embd)
    params = []
    for mat in state_dict.values():
        for row in mat:
            for p in row:
                params.append(p)
print(f"num params: {len(params)}")

# -----------------------------------------------------------------------------
# Model primitives
# -----------------------------------------------------------------------------


def linear(x, w):
    """Matrix-vector multiply: out[i] = sum_j w[i][j] * x[j]. Each row of w is a weight vector."""
    out = []
    for row in w:
        out.append(sum(wi * xi for wi, xi in zip(row, x)))
    return out


def softmax(logits):
    """Softmax with numerical stability: subtract max before exp to avoid overflow."""
    max_val = max(v.data for v in logits)
    exps = [(v - max_val).exp() for v in logits]
    total = sum(exps)
    return [e / total for e in exps]


def rmsnorm(x):
    """RMS normalization: scale each element by 1/sqrt(mean(x^2) + eps). No learnable scale/bias."""
    mean_sq = sum(xi * xi for xi in x) / len(x)
    scale = (mean_sq + 1e-5) ** -0.5
    return [xi * scale for xi in x]


def gpt(token_id, pos_id, keys, values):
    """
    Forward pass for one position. Causal: only sees tokens 0..pos_id via keys/values cache.
    Returns logits over next token (vocab_size values).
    """
    # Input: token + position embeddings, then normalize
    tok_emb = state_dict['wte'][token_id]
    pos_emb = state_dict['wpe'][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)

    for li in range(n_layer):
        # --- Attention block (pre-norm, residual) ---
        x_residual = x
        x = rmsnorm(x)

        # Q, K, V projections
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        keys[li].append(k)
        values[li].append(v)

        # Multi-head attention: each head does scaled dot-product attention
        x_attn = []
        for h in range(n_head):
            # Slice this head's Q, K, V from the full vectors
            hs = h * head_dim
            he = hs + head_dim
            q_h = q[hs:he]
            k_h = [ki[hs:he] for ki in keys[li]]
            v_h = [vi[hs:he] for vi in values[li]]

            # Attention scores: Q @ K^T / sqrt(d_k). Scaled to avoid softmax saturation.
            attn_logits = []
            for t in range(len(k_h)):
                score = sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / (head_dim ** 0.5)
                attn_logits.append(score)
            attn_weights = softmax(attn_logits)

            # Weighted sum of values: out = softmax(scores) @ V
            head_out = []
            for j in range(head_dim):
                head_out.append(sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))))
            x_attn.extend(head_out)

        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]

        # --- MLP block (pre-norm, residual) ---
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]

    logits = linear(x, state_dict['lm_head'])
    return logits

# -----------------------------------------------------------------------------
# Training: Adam optimizer with linear LR decay
# -----------------------------------------------------------------------------

learning_rate = 0.01
beta1 = 0.85   # momentum for first moment
beta2 = 0.99   # momentum for second moment (variance)
eps_adam = 1e-8  # numerical stability in denominator

m = [0.0] * len(params)  # first moment (biased estimate of gradient mean)
v = [0.0] * len(params)  # second moment (biased estimate of gradient variance)

num_steps = 1000

if not checkpoint:
    for step in range(num_steps):
        # --- Sample and tokenize one document ---
        doc = docs[step % len(docs)]
        tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
        n = min(block_size, len(tokens) - 1)  # number of next-token predictions

        # --- Forward pass: predict next token at each position ---
        keys = [[] for _ in range(n_layer)]
        values = [[] for _ in range(n_layer)]
        losses = []
        for pos_id in range(n):
            token_id = tokens[pos_id]
            target_id = tokens[pos_id + 1]
            logits = gpt(token_id, pos_id, keys, values)
            probs = softmax(logits)
            # Cross-entropy loss for this position: -log P(target)
            loss_t = -probs[target_id].log()
            losses.append(loss_t)
        loss = (1 / n) * sum(losses)

        # --- Backward pass: compute gradients ---
        loss.backward()

        # --- Adam update ---
        # Linear decay: lr goes from learning_rate down to 0 over num_steps
        lr_t = learning_rate * (1 - step / num_steps)
        for i, p in enumerate(params):
            g = p.grad
            # Update biased first and second moment estimates
            m[i] = beta1 * m[i] + (1 - beta1) * g
            v[i] = beta2 * v[i] + (1 - beta2) * (g ** 2)
            # Bias correction (Adam paper, Algorithm 1)
            m_hat = m[i] / (1 - beta1 ** (step + 1))
            v_hat = v[i] / (1 - beta2 ** (step + 1))
            # Parameter update: p -= lr * m_hat / (sqrt(v_hat) + eps)
            p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
            p.grad = 0

        print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}")

    save_checkpoint(uchars, state_dict)
    print(f"saved checkpoint to {CHECKPOINT_PATH}")

# -----------------------------------------------------------------------------
# Inference: autoregressive sampling
# -----------------------------------------------------------------------------

temperature = 0.5  # (0, 1]: lower = more deterministic, higher = more random
print("\n--- inference (new, hallucinated names) ---")

for sample_idx in range(20):
    keys = [[] for _ in range(n_layer)]
    values = [[] for _ in range(n_layer)]
    token_id = BOS
    sample = []
    for pos_id in range(block_size):
        logits = gpt(token_id, pos_id, keys, values)
        # Temperature scaling: divide logits before softmax to sharpen (low T) or soften (high T)
        scaled_logits = [l / temperature for l in logits]
        probs = softmax(scaled_logits)
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token_id == BOS:
            break
        sample.append(uchars[token_id])
    print(f"sample {sample_idx+1:2d}: {''.join(sample)}")