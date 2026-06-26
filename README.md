# NOWM: Neural Operator World Model

A world model for model-based reinforcement learning that formalises environment
dynamics as a **PDE on a spatial latent field**, solved by a learned Neural
Operator. Evaluated on the Atari 100k benchmark.

---

## Motivation

### Environments as PDEs

A world model must predict how the environment evolves. We formalise this as a
controlled stochastic PDE on a spatial domain Ω ⊂ ℝ²:

```
∂s/∂t = F_a[s] + η(x, t)
```

where `s(x, t) : Ω → ℝ^C` is a **vector-valued field** over space, `F_a` is the
**tendency operator** conditioned on action `a` (the physics of the environment),
and `η(x, t)` is a stochastic forcing field encoding aleatoric uncertainty and
unobserved dynamics.

This perspective is natural. Pixels in a game frame are samples of a spatial
field: `s(x, t)` represents latent quantities at each 2D location — object
identity, velocity, material properties. The dynamics are local or semi-local: a
ball moves according to rules that operate on its neighbourhood. Treating the
state as a function rather than a vector lets the model respect this structure.

### From PDEs to Operator Learning

The classical approach discretises the PDE analytically. Here, `F_a` is
**unknown** and must be learned from data. The key insight is that `F_a` is an
**operator** — a map from one function space to another:

```
F_a : (Ω → ℝ^C) → (Ω → ℝ^C)
```

Neural Operators are a class of architectures designed to learn such maps.
Unlike standard networks that learn finite-dimensional functions, Neural
Operators learn mappings that are discretisation-invariant and can generalise
across spatial resolutions. Concretely, the learned operator takes the form:

```
(F_a[s])(x) = σ( W s(x) + (K_a * s)(x) )
```

where `W` is a pointwise linear transform and `K_a` is a global integral kernel
conditioned on `a`. Different parameterisations of `K_a` give different operator
families: spectral (FNO), wavelet (WNO), or attention-based (AttnNO).

### Stochastic Forcing and Data Assimilation

The stochastic term `η(x, t)` is modelled as a latent variable inferred from
observations — the standard **data assimilation** setup from computational
geophysics. At each timestep:

1. **Predict** — advance the state field forward through the operator: `ŝ_{t+1} = F_a[s_t]`
2. **Assimilate** — incorporate the new observation to correct the estimate: `s_{t+1} = ŝ_{t+1} + Δ(o_{t+1})`

This two-step cycle is exactly the NOWM observe loop: `_core` (prediction) and
`_spatial_observe` (assimilation). The stochastic posterior over `η` is the
standard ELBO objective familiar from variational world models.

---

## Architecture

### Latent State

The world model carry is split into two components that are concatenated for
compatibility with downstream heads:

```
carry['deter'] = concat(flatten(h_spatial), h_vec)
                  (B, H×W×C)             (B, D)
```

| Component | Shape | Role |
|---|---|---|
| `h_spatial` | `B × H × W × C` | 2D spatial dynamics field |
| `h_vec` | `B × D` | Global summary state (GRU) |
| `stoch` | `B × N × K` | Categorical stochastic latent |

Default Atari 100k config: `H=W=8`, `C=32`, `D=512`, `N=32`, `K=64`.

### Dynamics (`_core`)

One step of pure imagination (no observation):

```
h_s, h_v  ←  split(deter)

ctx       =  act(norm( Linear(action) + Linear(h_v) ))   # action + global context

h_s_mixed =  AttnNO(norm(h_s), ctx)                      # spatial operator
h_s_cand  =  act(norm( h_s_mixed + stoch_broadcast ))     # stochastic forcing
h_s_new   =  GateUpdate(h_s_cand, h_s)                    # GRU-style gate

s2g       =  MultiheadAttn(q=h_v, kv=h_s_new)             # spatial → global readout
h_v_new   =  GRU(h_v, concat(s2g, action, stoch))

deter_new =  concat(flatten(h_s_new), h_v_new)
```

**Spatial operator (`AttnNO`):** Transformer with 2D Rotary Position Embedding
(RoPE), per-head QK-Norm, and FiLM conditioning on `ctx`. Operates on the
`H×W` token grid, learning arbitrary long-range spatial interactions.

**s2g attention:** `h_vec` acts as a query into `h_spatial`, letting the global
state actively read out relevant spatial locations (e.g. ball position, nearby
hazard). The raw (unnormalized) spatial field is used as key/value so that
magnitude carries spatial saliency information.

Alternative spatial operators can be selected via `spatial_op`:

| Value | Operator | Characteristics |
|---|---|---|
| `attnno` | Transformer + 2D RoPE | Arbitrary spatial relations; default |
| `fno` | Fourier Neural Operator | Global spectral mixing; smooth PDEs |
| `wno` | Wavelet Neural Operator | Multi-scale Haar; localized objects |
| `conv` | 3×3 convolution | Local motion; fastest |

### Observation Assimilation (`_spatial_observe` + `_vec_observe`)

When an observation arrives, both state components are updated:

**Spatial assimilation:**
```
enc_tok  =  CNN(obs)                    # (B, H×W, C_enc)
vel      =  proj(enc_tok) − proj(enc_tok_prev)   # per-location motion delta

h_s_in  =  h_s + Linear(enc_tok) + Linear(vel)
h_s_cand =  AttnNO(norm(h_s_in), ctx_obs)
h_s_post =  GateUpdate(h_s_cand, h_s)
```

The velocity signal `vel` encodes direction and speed rather than appearance,
providing a physics-grounded motion prior without leaking raw pixel values.

**Global assimilation:**
```
enc_attended  =  MultiheadAttn(q=h_v, kv=enc_tok)   # h_vec selects relevant region
h_v_post      =  GRU(h_v, enc_attended)
```

### Encoder

A CNN that downsamples the `64×64` input to the `H×W` spatial grid:

```
3 conv layers (each followed by 2× maxpool):  64×64 → 32×32 → 16×16 → 8×8
output channels = depth × mults[-1]
```

Current Atari 100k config: `depth=32`, `mults=[1,1,1]` → **32 channels per
spatial position**, aligned with `lat_chan=32`. This eliminates the 8:1 compression
bottleneck that existed when the encoder output (256 ch) was projected down to
`lat_chan` in a single linear layer.

### Decoder

The decoder uses `h_spatial` directly as a starting feature map and upsamples
with transposed convolutions back to `64×64`:

```
h_spatial (8×8×32)  →  FiLM(h_vec)  →  Upsample×3  →  64×64×3
```

FiLM conditioning from `h_vec` lets the global state modulate every spatial
location during reconstruction.

### Stochastic Latent

Global categorical stochastic variable (`stoch_spatial=False`, default for
Atari 100k):

- Prior: `MLP(h_spatial, h_vec)` with per-location `h_vec` FiLM conditioning
- Posterior: same network conditioned additionally on `enc_tok`
- KL trained with `free_nats=0.5` as a floor on the summed KL

---

## Key Differences from DreamerV3

| | DreamerV3 | NOWM |
|---|---|---|
| State | Flat RSSM vector | 2D spatial field + global GRU |
| Dynamics | GRU over flat state | Neural Operator over spatial grid |
| Spatial reasoning | Implicit | Explicit 2D structure |
| Decoder start | Flat broadcast | `h_spatial` feature map directly |
| Encoder channels | 256 (8× lat_chan) | 32 (= lat_chan, no compression) |
| Position encoding | None | 2D RoPE inside AttnNO |

---

## Policy Training

The actor-critic policy is trained on imagined rollouts from the world model
(DreamerV3 style). Key design choices:

**Soft entropy gate:** Instead of a binary entropy floor gate that blocks all
policy gradient when `rand < min_rand` (causing deadlock when entropy collapses),
NOWM uses a linear ramp:

```python
pol_gate = clip(rand / min_rand, 0, 1)
```

This scales the policy gradient proportional to the current entropy level,
preventing complete gradient cutoff while still protecting against low-entropy
overconfident updates.

**Adaptive entropy coefficient (`actent_adapt`):** SAC-style automatic tuning
of the entropy coefficient toward `target_rand=0.3`, preventing both entropy
collapse and excessive randomness.

---

## Running

### Atari 100k (single game)

```sh
bash train_atari100k.sh pong
```

### Atari 100k (all 26 games)

```sh
bash train_atari100k.sh
```

### Custom options

```sh
GPU=1 SEED=1 LOGROOT=logs/my_run bash train_atari100k.sh pong
```

### Direct invocation

```sh
python -m dreamerv3.main \
  --configs atari100k nowm \
  --task atari100k_pong \
  --seed 0 \
  --logdir logs/pong_s0
```

### Viewing results

```sh
pip install -U scope
python -m scope.viewer --basedir logs --port 8000
```

Metrics are also written as JSONL files in the log directory.

---

## Configuration

Key parameters under `atari100k` in `dreamerv3/configs.yaml`:

| Parameter | Default | Description |
|---|---|---|
| `dyn.nowm.lat_size` | `8` | Spatial grid side length (H=W) |
| `dyn.nowm.lat_chan` | `32` | Channels per spatial position |
| `dyn.nowm.deter` | `512` | Global h_vec dimension |
| `dyn.nowm.stoch` | `32` | Number of stochastic categories |
| `dyn.nowm.classes` | `64` | Classes per category |
| `dyn.nowm.spatial_op` | `attnno` | Spatial operator type |
| `dyn.nowm.attn_heads` | `8` | Attention heads in AttnNO |
| `enc.simple.depth` | `32` | Encoder base channel width |
| `enc.simple.mults` | `[1,1,1]` | Per-level channel multipliers |
| `agent.imag_last` | `0` | Imagination start diversity (0 = any step) |
| `agent.imag_length` | `25` | Imagined rollout horizon |
| `agent.imag_loss.target_rand` | `0.3` | Target entropy fraction |
| `agent.imag_loss.actent_rate` | `3e-3` | Entropy adaptation speed |
| `agent.imag_loss.min_rand` | `0.05` | Soft gate activation threshold |

---

## Requirements

Python 3.11+, JAX with CUDA support.

```sh
pip install -U -r requirements.txt
```

The `debug` config block reduces all dimensions for fast CPU debugging:

```sh
python -m dreamerv3.main --configs atari100k nowm debug --task atari100k_pong --jax.platform cpu
```
