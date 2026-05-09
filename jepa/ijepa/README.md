### I-JEPA from Scratch: Learning Curriculum

### 🔧 Tiny Model Spec (we'll use this throughout)

| Parameter | Value | Why |
| --- | --- | --- |
| Image size | 16×16, 1 channel | Tiny but tractable |
| Patch size | 4×4 | → 4×4 = **16 patches** total |
| Embed dim `D` | 16 | Easy to print and inspect |
| Transformer layers | 2 | Deep enough to be real |
| Attention heads | 2 | D/heads = 8 per head |
| MLP hidden dim | 32 | Standard 2× expansion |
| Batch size | 2 | See batching behavior |

At every step you'll track tensor shapes explicitly, like a signal trace.

---

### 📚 Module Outline

---

**Module 1 — Patch Extraction & Embedding**

> *"Chop the image into tokens"*
> 
- Split `(B, C, H, W)` image into `(B, N, P²·C)` patches where  $N = \frac{H}{P} \cdot \frac{W}{P}$
- Apply a linear projection: $z=xW_e+b_e$ where $W_e \in \mathbb{R}^{P^2 C \times D}$
- 🧪 **Exercise**: `patch_embed(x, patch_size=4, embed_dim=16)` → shape `(2, 16, 16)`

---

**Module 2 — Positional Encoding**

> *"Give each token a grid address"*
> 
- 2D sinusoidal encoding so the model knows *where* each patch lives
- 🧪 **Exercise**: `positional_encoding_2d(grid_h=4, grid_w=4, embed_dim=16)` → shape `(16, 16)`, add to patch embeddings

---

**Module 3 — Scaled Dot-Product Attention**

> *"The core compute primitive"*
> 

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

- 🧪 **Exercise**: `scaled_dot_product_attention(Q, K, V)` from raw numpy — inspect the attention weight matrix

---

**Module 4 — Multi-Head Self-Attention + Transformer Block**

> *"Stack the primitive into a real layer"*
> 
- Split $D$ into $h$ heads, run attention in parallel, concat + project
- Add LayerNorm, residual connections, and an MLP sublayer
- 🧪 **Exercise**: `TransformerBlock` as a class with a `forward(x)` method; shape in = shape out = `(B, N, D)`

---

**Module 5 — Masking Strategy**

> *"The I-JEPA-specific trick"*
> 
- Sample **target blocks**: 4 random rectangular regions of patches (indices)
- **Context** = all patches **minus** the target patches
- 🧪 **Exercise**: `sample_masks(grid_h, grid_w)` → return `context_indices`, `target_indices`; visualize which patches are which

---

**Module 6 — Context Encoder & Target Encoder**

> *"Two encoders, one frozen"*
> 
- **Context encoder**: runs only on context patch tokens
- **Target encoder**: runs on *all* patches, but only its outputs at target positions matter
- Target encoder = EMA copy of context encoder (weights are **not** updated by backprop)
- 🧪 **Exercise**: forward pass through both; compare output shapes

---

**Module 7 — Predictor**

> *"Bridge the gap"*
> 
- Small transformer that takes context encoder output + **learnable mask tokens** placed at target positions
- Predicts representations at each target position
- 🧪 **Exercise**: `Predictor.forward(context_encoding, target_indices)` → shape `(B, N_target, D)`

---

**Module 8 — Loss**

> *"What we're minimizing"*
> 

$$
\mathcal{L} = \frac{1}{|\mathcal{T}|} \sum_{t \in \mathcal{T}} \left\| \hat{z}_t - \text{sg}(z_t) \right\|_2^2
$$

- L2 between predictor output and target encoder output
- `sg(·)` = stop-gradient (just `detach` / don't backprop through it)
- 🧪 **Exercise**: implement `ijepa_loss(predicted, targets)`

---

**Module 9 — EMA Weight Update**

> *"How the target encoder stays stable"*
> 

$$
\theta_{\text{target}} \leftarrow \tau \cdot \theta_{\text{target}} + (1 - \tau) \cdot \theta_{\text{context}}
$$

- 🧪 **Exercise**: `ema_update(context_params, target_params, tau=0.996)`

---

**Module 10 — Full Training Loop**

> *"Put it all together"*
> 
- Wire up Modules 1–9 into a single `train_step(image_batch)`
- Run 50 steps on random images, watch the loss curve
- 🧪 **Exercise**: annotate every intermediate tensor shape in your loop as a comment

---

### How we'll work through these

Each module will be:

1. A brief explanation of *what* is being computed and *why*
2. The math (in LaTeX)
3. A skeleton function for you to fill in
4. A shape-check test you run to verify correctness
5. A quick question to make sure the concept clicked before moving on