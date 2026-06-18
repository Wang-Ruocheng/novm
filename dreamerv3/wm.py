import math

import einops
import elements
import embodied.jax
import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

f32 = jnp.float32
sg = jax.lax.stop_gradient


class Encoder(nj.Module):

  units: int = 1024
  norm: str = 'rms'
  act: str = 'gelu'
  depth: int = 64
  mults: tuple = (2, 3, 4, 4)
  layers: int = 3
  kernel: int = 5
  symlog: bool = True
  outer: bool = False
  strided: bool = False

  def __init__(self, obs_space, **kw):
    assert all(len(s.shape) <= 3 for s in obs_space.values()), obs_space
    self.obs_space = obs_space
    self.veckeys = [k for k, s in obs_space.items() if len(s.shape) <= 2]
    self.imgkeys = [k for k, s in obs_space.items() if len(s.shape) == 3]
    self.depths = tuple(self.depth * mult for mult in self.mults)
    self.kw = kw

  @property
  def entry_space(self):
    return {}

  def initial(self, batch_size):
    return {}

  def truncate(self, entries, carry=None):
    return {}

  def __call__(self, carry, obs, reset, training, single=False):
    bdims = 1 if single else 2
    outs = []
    bshape = reset.shape

    if self.veckeys:
      vspace = {k: self.obs_space[k] for k in self.veckeys}
      vecs = {k: obs[k] for k in self.veckeys}
      squish = nn.symlog if self.symlog else lambda x: x
      x = nn.DictConcat(vspace, 1, squish=squish)(vecs)
      x = x.reshape((-1, *x.shape[bdims:]))
      for i in range(self.layers):
        x = self.sub(f'mlp{i}', nn.Linear, self.units, **self.kw)(x)
        x = nn.act(self.act)(self.sub(f'mlp{i}norm', nn.Norm, self.norm)(x))
      outs.append(x)

    if self.imgkeys:
      K = self.kernel
      imgs = [obs[k] for k in sorted(self.imgkeys)]
      assert all(x.dtype == jnp.uint8 for x in imgs)
      x = nn.cast(jnp.concatenate(imgs, -1), force=True) / 255 - 0.5
      x = x.reshape((-1, *x.shape[bdims:]))
      for i, depth in enumerate(self.depths):
        if self.outer and i == 0:
          x = self.sub(f'cnn{i}', nn.Conv2D, depth, K, **self.kw)(x)
        elif self.strided:
          x = self.sub(f'cnn{i}', nn.Conv2D, depth, K, 2, **self.kw)(x)
        else:
          x = self.sub(f'cnn{i}', nn.Conv2D, depth, K, **self.kw)(x)
          B, H, W, C = x.shape
          x = x.reshape((B, H // 2, 2, W // 2, 2, C)).max((2, 4))
        x = nn.act(self.act)(self.sub(f'cnn{i}norm', nn.Norm, self.norm)(x))
      assert 3 <= x.shape[-3] <= 16, x.shape
      assert 3 <= x.shape[-2] <= 16, x.shape
      x = x.reshape((x.shape[0], -1))
      outs.append(x)

    x = jnp.concatenate(outs, -1)
    tokens = x.reshape((*bshape, *x.shape[1:]))
    entries = {}
    return carry, entries, tokens


class Decoder(nj.Module):

  units: int = 1024
  norm: str = 'rms'
  act: str = 'gelu'
  outscale: float = 1.0
  depth: int = 64
  mults: tuple = (2, 3, 4, 4)
  layers: int = 3
  kernel: int = 5
  symlog: bool = True
  bspace: int = 8
  outer: bool = False
  strided: bool = False

  def __init__(self, obs_space, **kw):
    assert all(len(s.shape) <= 3 for s in obs_space.values()), obs_space
    self.obs_space = obs_space
    self.veckeys = [k for k, s in obs_space.items() if len(s.shape) <= 2]
    self.imgkeys = [k for k, s in obs_space.items() if len(s.shape) == 3]
    self.depths = tuple(self.depth * mult for mult in self.mults)
    self.imgdep = sum(obs_space[k].shape[-1] for k in self.imgkeys)
    self.imgres = self.imgkeys and obs_space[self.imgkeys[0]].shape[:-1]
    self.kw = kw

  @property
  def entry_space(self):
    return {}

  def initial(self, batch_size):
    return {}

  def truncate(self, entries, carry=None):
    return {}

  def __call__(self, carry, feat, reset, training, single=False):
    assert feat['deter'].shape[-1] % self.bspace == 0
    K = self.kernel
    recons = {}
    bshape = reset.shape
    inp = [nn.cast(feat[k]) for k in ('stoch', 'deter')]
    inp = [x.reshape((math.prod(bshape), -1)) for x in inp]
    inp = jnp.concatenate(inp, -1)

    if self.veckeys:
      spaces = {k: self.obs_space[k] for k in self.veckeys}
      o1, o2 = 'categorical', ('symlog_mse' if self.symlog else 'mse')
      outputs = {k: o1 if v.discrete else o2 for k, v in spaces.items()}
      kw = dict(**self.kw, act=self.act, norm=self.norm)
      x = self.sub('mlp', nn.MLP, self.layers, self.units, **kw)(inp)
      x = x.reshape((*bshape, *x.shape[1:]))
      kw = dict(**self.kw, outscale=self.outscale)
      outs = self.sub('vec', embodied.jax.DictHead, spaces, outputs, **kw)(x)
      recons.update(outs)

    if self.imgkeys:
      factor = 2 ** (len(self.depths) - int(bool(self.outer)))
      minres = [int(x // factor) for x in self.imgres]
      assert 3 <= minres[0] <= 16, minres
      assert 3 <= minres[1] <= 16, minres
      shape = (*minres, self.depths[-1])
      if self.bspace:
        u, g = math.prod(shape), self.bspace
        x0, x1 = nn.cast((feat['deter'], feat['stoch']))
        x1 = x1.reshape((*x1.shape[:-2], -1))
        x0 = x0.reshape((-1, x0.shape[-1]))
        x1 = x1.reshape((-1, x1.shape[-1]))
        x0 = self.sub('sp0', nn.BlockLinear, u, g, **self.kw)(x0)
        x0 = einops.rearrange(
            x0, '... (g h w c) -> ... h w (g c)',
            h=minres[0], w=minres[1], g=g)
        x1 = self.sub('sp1', nn.Linear, 2 * self.units, **self.kw)(x1)
        x1 = nn.act(self.act)(self.sub('sp1norm', nn.Norm, self.norm)(x1))
        x1 = self.sub('sp2', nn.Linear, shape, **self.kw)(x1)
        x = nn.act(self.act)(self.sub('spnorm', nn.Norm, self.norm)(x0 + x1))
      else:
        x = self.sub('space', nn.Linear, shape, **kw)(inp)
        x = nn.act(self.act)(self.sub('spacenorm', nn.Norm, self.norm)(x))
      for i, depth in reversed(list(enumerate(self.depths[:-1]))):
        if self.strided:
          kw = dict(**self.kw, transp=True)
          x = self.sub(f'conv{i}', nn.Conv2D, depth, K, 2, **kw)(x)
        else:
          x = x.repeat(2, -2).repeat(2, -3)
          x = self.sub(f'conv{i}', nn.Conv2D, depth, K, **self.kw)(x)
        x = nn.act(self.act)(self.sub(f'conv{i}norm', nn.Norm, self.norm)(x))
      if self.outer:
        kw = dict(**self.kw, outscale=self.outscale)
        x = self.sub('imgout', nn.Conv2D, self.imgdep, K, **kw)(x)
      elif self.strided:
        kw = dict(**self.kw, outscale=self.outscale, transp=True)
        x = self.sub('imgout', nn.Conv2D, self.imgdep, K, 2, **kw)(x)
      else:
        x = x.repeat(2, -2).repeat(2, -3)
        kw = dict(**self.kw, outscale=self.outscale)
        x = self.sub('imgout', nn.Conv2D, self.imgdep, K, **kw)(x)
      x = jax.nn.sigmoid(x)
      x = x.reshape((*bshape, *x.shape[1:]))
      split = np.cumsum(
          [self.obs_space[k].shape[-1] for k in self.imgkeys][:-1])
      for k, out in zip(self.imgkeys, jnp.split(x, split, -1)):
        out = embodied.jax.outs.MSE(out)
        out = embodied.jax.outs.Agg(out, 3, jnp.sum)
        recons[k] = out

    entries = {}
    return carry, entries, recons


class NOWM(nj.Module):
  """Neural Operator World Model: 2D spatial h_spatial + global h_vec.

  carry['deter'] = concat(flatten(h_spatial), h_vec)  — flat for downstream
  h_spatial (B, lat_size, lat_size, lat_chan): spatial dynamics via spatial_op
  h_vec     (B, deter): GRU global state

  spatial_op choices:
    'wno'  — Wavelet Neural Operator (multi-scale Haar, localized objects)
    'fno'  — Fourier Neural Operator (global spectral mixing, smooth PDEs)
    'attn' — local self-attention (arbitrary spatial relations, discrete objects)
    'conv' — 3×3 local convolution (local motion, fast)
  """

  # Global vector state h_vec
  deter: int = 512
  hidden: int = 256
  stoch: int = 32
  classes: int = 32
  norm: str = 'rms'
  act: str = 'gelu'
  unroll: bool = False
  unimix: float = 0.01
  outscale: float = 1.0
  imglayers: int = 2
  obslayers: int = 1
  free_nats: float = 1.0
  absolute: bool = False
  # Spatial field h_spatial (lat_size × lat_size × lat_chan)
  lat_size: int = 8
  lat_chan: int = 32
  # Spatial operator: 'fno' | 'attn' | 'conv'
  spatial_op: str = 'attn'
  attn_heads: int = 4        # used when spatial_op='attn'
  fno_modes: int = 4         # used when spatial_op='fno'
  # Spatial stochastic latent field: each location gets its own stoch variables.
  # Enables per-location posterior and per-location stochastic PDE forcing.
  # stoch_total = lat_size^2 * stoch; stoch/classes are now per-location counts.
  stoch_spatial: bool = True

  def __init__(self, act_space, **kw):
    self.act_space = act_space
    self.kw = kw

  def _sp(self):
    return self.lat_size * self.lat_size * self.lat_chan

  def _stoch_total(self):
    """Total stoch variables: H*W*stoch (spatial) or stoch (global)."""
    if self.stoch_spatial:
      return self.lat_size * self.lat_size * self.stoch
    return self.stoch

  def _total(self):
    return self._sp() + self.deter

  @property
  def entry_space(self):
    return dict(
        deter=elements.Space(np.float32, self._total()),
        stoch=elements.Space(np.float32, (self._stoch_total(), self.classes)))

  def initial(self, bsize):
    return nn.cast(dict(
        deter=jnp.zeros([bsize, self._total()], f32),
        stoch=jnp.zeros([bsize, self._stoch_total(), self.classes], f32)))

  def truncate(self, entries, carry=None):
    assert entries['deter'].ndim == 3, entries['deter'].shape
    return jax.tree.map(lambda x: x[:, -1], entries)

  def starts(self, entries, carry, nlast):
    B = len(jax.tree.leaves(carry)[0])
    return jax.tree.map(
        lambda x: x[:, -nlast:].reshape((B * nlast, *x.shape[2:])), entries)

  def observe(self, carry, tokens, action, reset, training, single=False):
    carry, tokens, action = nn.cast((carry, tokens, action))
    if single:
      carry, (entry, feat) = self._observe(carry, tokens, action, reset, training)
      return carry, entry, feat
    else:
      unroll = jax.tree.leaves(tokens)[0].shape[1] if self.unroll else 1
      carry, (entries, feat) = nj.scan(
          lambda carry, inputs: self._observe(carry, *inputs, training),
          carry, (tokens, action, reset), unroll=unroll, axis=1)
      return carry, entries, feat

  def _observe(self, carry, tokens, action, reset, training):
    deter, stoch, action = nn.mask(
        (carry['deter'], carry['stoch'], action), ~reset)
    action = nn.DictConcat(self.act_space, 1)(action)
    action = nn.mask(action, ~reset)
    deter = self._core(deter, stoch, action)
    tokens = tokens.reshape((*deter.shape[:-1], -1))
    deter_prior = deter                              # prior deter: no observation info
    deter = self._spatial_observe(deter, tokens)    # update h_spatial per-location
    deter = self._vec_observe(deter, tokens)        # update h_vec from global pool
    # obslogit: per-location (spatial stoch) or global, using prior_deter+tokens.
    # prior_deter is the ONLY input that differs from prior → stoch must carry obs.
    if self.stoch_spatial:
      logit = self._obs_logit_spatial(deter_prior, tokens)
    else:
      x = tokens if self.absolute else jnp.concatenate([deter_prior, tokens], -1)
      for i in range(self.obslayers):
        x = self.sub(f'obs{i}', nn.Linear, self.hidden, **self.kw)(x)
        x = nn.act(self.act)(self.sub(f'obs{i}norm', nn.Norm, self.norm)(x))
      logit = self._logit('obslogit', x)
    stoch = nn.cast(self._dist(logit).sample(seed=nj.seed()))
    # carry uses prior_deter: past obs enter only through the stoch bottleneck,
    # forcing the decoder to rely on current stoch → KL stays non-zero.
    # entry keeps posterior_deter so imagination starts from spatially-corrected states.
    carry = dict(deter=deter_prior, stoch=stoch)
    feat = dict(deter=deter_prior, prior_deter=deter_prior, stoch=stoch, logit=logit)
    entry = dict(deter=deter, stoch=stoch)
    assert all(x.dtype == nn.COMPUTE_DTYPE for x in (deter, stoch, logit))
    return carry, (entry, feat)

  def _spatial_observe(self, deter, tokens):
    """Per-location posterior: update h_spatial[i,j] using enc_spatial[i,j].

    Encoder CNN preserves spatial layout, so enc[i,j] aligns with h_s[i,j].
    We fuse them per-location with a gated MLP, then write back into deter.
    """
    H = W = self.lat_size
    C = self.lat_chan
    B = deter.shape[0]
    sp = self._sp()
    h_s_tok = deter[:, :sp].reshape(B, H * W, C)
    h_v = deter[:, sp:]

    # Infer encoder channel dim; works for pure-image envs (atari100k, crafter)
    assert tokens.shape[-1] % (H * W) == 0, (
        f'Token dim {tokens.shape[-1]} not divisible by H*W={H*W}. '
        f'Check enc.simple.mults and lat_size match.')
    C_enc = tokens.shape[-1] // (H * W)
    enc_tok = tokens.reshape(B, H * W, C_enc)       # (B, HW, C_enc)

    # Per-location fusion
    x = jnp.concatenate([h_s_tok, enc_tok], axis=-1)  # (B, HW, C+C_enc)
    for i in range(self.obslayers):
      x = self.sub(f'spa_obs{i}', nn.Linear, self.hidden, **self.kw)(x)
      x = nn.act(self.act)(self.sub(f'spa_obs{i}norm', nn.Norm, self.norm)(x))

    # Gated update (bias=-1 → near-identity at init, stable training start)
    gate = jax.nn.sigmoid(
        self.sub('spa_gate', nn.Linear, C, **self.kw)(
            jnp.concatenate([h_s_tok, x], axis=-1)) - 1)
    delta = self.sub('spa_proj', nn.Linear, C, **self.kw)(x)
    h_s_post = gate * jnp.tanh(delta) + (1 - gate) * h_s_tok  # (B, HW, C)

    return jnp.concatenate([h_s_post.reshape(B, sp), h_v], axis=-1)

  def _vec_observe(self, deter, tokens):
    """Global posterior: update h_vec from globally-pooled encoder tokens.

    Encoder spatial tokens are averaged across HW positions, giving a single
    global feature that captures non-spatial content (score, HUD, etc.).
    This feeds directly into h_vec so non-spatial info doesn't pollute h_spatial.
    """
    sp = self._sp()
    D = self.deter
    B = deter.shape[0]
    H = W = self.lat_size

    h_v = deter[:, sp:]  # (B, D)

    C_enc = tokens.shape[-1] // (H * W)
    enc_tok = tokens.reshape(B, H * W, C_enc)   # (B, HW, C_enc)
    enc_global = jnp.concatenate(
        [enc_tok.mean(axis=1), enc_tok.max(axis=1)], axis=-1)  # (B, 2*C_enc)

    x = jnp.concatenate([h_v, enc_global], axis=-1)
    for i in range(self.obslayers):
      x = self.sub(f'vec_obs{i}', nn.Linear, self.hidden, **self.kw)(x)
      x = nn.act(self.act)(self.sub(f'vec_obs{i}norm', nn.Norm, self.norm)(x))

    gate = jax.nn.sigmoid(
        self.sub('vec_gate', nn.Linear, D, **self.kw)(
            jnp.concatenate([h_v, x], axis=-1)) - 1)
    delta = self.sub('vec_proj', nn.Linear, D, **self.kw)(x)
    h_v_post = gate * jnp.tanh(delta) + (1 - gate) * h_v  # (B, D)

    return jnp.concatenate([deter[:, :sp], h_v_post], axis=-1)

  def imagine(self, carry, policy, length, training, single=False):
    if single:
      action = policy(sg(carry)) if callable(policy) else policy
      actemb = nn.DictConcat(self.act_space, 1)(action)
      deter = self._core(carry['deter'], carry['stoch'], actemb)
      logit = self._prior(deter)
      stoch = nn.cast(self._dist(logit).sample(seed=nj.seed()))
      carry = nn.cast(dict(deter=deter, stoch=stoch))
      feat = nn.cast(dict(deter=deter, prior_deter=deter, stoch=stoch, logit=logit))
      assert all(x.dtype == nn.COMPUTE_DTYPE for x in (deter, stoch, logit))
      return carry, (feat, action)
    else:
      unroll = length if self.unroll else 1
      if callable(policy):
        carry, (feat, action) = nj.scan(
            lambda c, _: self.imagine(c, policy, 1, training, single=True),
            nn.cast(carry), (), length, unroll=unroll, axis=1)
      else:
        carry, (feat, action) = nj.scan(
            lambda c, a: self.imagine(c, a, 1, training, single=True),
            nn.cast(carry), nn.cast(policy), length, unroll=unroll, axis=1)
      return carry, feat, action

  def loss(self, carry, tokens, acts, reset, training):
    metrics = {}
    carry, entries, feat = self.observe(carry, tokens, acts, reset, training)
    prior = self._prior(feat['prior_deter'])  # must use prior deter, not posterior
    post = feat['logit']
    dyn = self._dist(sg(post)).kl(self._dist(prior))
    rep = self._dist(post).kl(self._dist(sg(prior)))
    if self.free_nats:
      fn = self.free_nats * self._stoch_total()  # per-variable → total floor
      dyn = jnp.maximum(dyn, fn)
      rep = jnp.maximum(rep, fn)
    losses = {'dyn': dyn, 'rep': rep}
    metrics['dyn_ent'] = self._dist(prior).entropy().mean()
    metrics['rep_ent'] = self._dist(post).entropy().mean()
    return carry, entries, losses, feat, metrics

  def _core(self, deter, stoch, action):
    H = W = self.lat_size
    C = self.lat_chan
    D = self.deter
    B = deter.shape[0]
    sp = self._sp()
    compute_dtype = deter.dtype
    stoch_flat = stoch.reshape(B, -1)
    action = action / sg(jnp.maximum(1, jnp.abs(action)))

    # Split deter → (h_spatial, h_vec)
    h_s = deter[:, :sp].reshape(B, H, W, C)   # (B, H, W, C)
    h_v = deter[:, sp:]                         # (B, D)
    h_s_tok = h_s.reshape(B, H * W, C)          # (B, HW, C) for attention

    # ---- Global → spatial mixing ----
    # h_v broadcasts a summary to all spatial locations so the spatial field
    # knows the global state before running FNO dynamics.
    h_v_proj = self.sub('mix_v2s', nn.Linear, C, **self.kw)(h_v)  # (B, C)
    h_s_tok = h_s_tok + h_v_proj[:, None, :]                       # (B, HW, C)
    h_s_tok = self.sub('mix_norm_s', nn.Norm, self.norm)(h_s_tok)

    # Action embedding — conditions the spatial operator via per-subband FiLM
    # For WNO: FiLM inside each wavelet subband (action drives the operator itself)
    # For fno/attn/conv: additive forcing fallback
    act_emb = self.sub('op_act', nn.Linear, C, **self.kw)(action)  # (B, C)

    # ---- Spatial operator: FNO / Attn / Conv / WNO (action-conditioned) ----
    h_s_mixed = self._spatial_op(h_s_tok, B, H, W, C, deter.dtype, act_emb)

    # Stochastic forcing: per-location (spatial) or broadcast (global)
    # Spatial: η(x,t) per location — stochastic PDE noise field
    # Global:  η(t) broadcast — same noise for every location
    if self.stoch_spatial:
      stoch_per_loc = stoch.reshape(B, H * W, self.stoch * self.classes)
      sto_f = self.sub('op_sto', nn.Linear, C, **self.kw)(stoch_per_loc)  # (B, HW, C)
      sto_f = nn.act(self.act)(self.sub('op_stonorm', nn.Norm, self.norm)(sto_f))
    else:
      sto_f = self.sub('op_sto', nn.Linear, C, **self.kw)(stoch_flat)     # (B, C)
      sto_f = nn.act(self.act)(self.sub('op_stonorm', nn.Norm, self.norm)(sto_f))
      sto_f = sto_f[:, None, :]                                            # broadcast

    # O_env(action-conditioned) + F_stoch
    cand_s_tok = h_s_mixed + sto_f
    cand_s_tok = nn.act(self.act)(
        self.sub('op_candnorm', nn.Norm, self.norm)(cand_s_tok))

    # Gated update for h_spatial (GRU-style stability)
    gate_s = jax.nn.sigmoid(
        self.sub('op_gate', nn.Linear, C, **self.kw)(
            jnp.concatenate([h_s_tok, cand_s_tok], -1)) - 1)
    h_s_new = (
        gate_s * jnp.tanh(cand_s_tok) + (1 - gate_s) * h_s_tok
    ).reshape(B, H, W, C)

    # ---- Spatial → global aggregation ----
    # Mean + max dual pooling, then project to hidden dim.
    # This gives h_vec a richer spatial summary (hidden=256) instead of the
    # raw 16-dim mean pool that was dominated by h_v (512 dims) in the GRU.
    h_s_tok_new = h_s_new.reshape(B, H * W, C)
    s2g = jnp.concatenate(
        [h_s_tok_new.mean(axis=1), h_s_tok_new.max(axis=1)], axis=-1)  # (B, 2C)
    s2g = self.sub('s2g_proj', nn.Linear, self.hidden, **self.kw)(s2g)
    s2g = nn.act(self.act)(self.sub('s2g_norm', nn.Norm, self.norm)(s2g))

    # ---- GRU update for h_vec ----
    gru_in = jnp.concatenate([h_v, s2g, action, stoch_flat], -1)
    gru_out = self.sub('vec_gru', nn.Linear, 3 * D, **self.kw)(gru_in)
    r, cand_v, upd = jnp.split(gru_out, 3, -1)
    r = jax.nn.sigmoid(r)
    cand_v = jnp.tanh(r * cand_v)
    upd = jax.nn.sigmoid(upd - 1)
    h_v_new = upd * cand_v + (1 - upd) * h_v

    return jnp.concatenate([h_s_new.reshape(B, sp), h_v_new], axis=-1)

  def _prior(self, feat):
    if self.stoch_spatial:
      H = W = self.lat_size; C = self.lat_chan; sp = self._sp()
      h_s_tok = feat[..., :sp].reshape(*feat.shape[:-1], H * W, C)  # (..., HW, C)
      h_v = feat[..., sp:]                                           # (..., D)
      h_v_proj = self.sub('prior_v2s', nn.Linear, C, **self.kw)(h_v)  # (..., C)
      h_s_tok = h_s_tok + h_v_proj[..., None, :]                    # (..., HW, C)
      x = h_s_tok
      for i in range(self.imglayers):
        x = self.sub(f'prior{i}', nn.Linear, self.hidden, **self.kw)(x)
        x = nn.act(self.act)(self.sub(f'prior{i}norm', nn.Norm, self.norm)(x))
      return self._spatial_logit('priorlogit', x)
    else:
      x = feat
      for i in range(self.imglayers):
        x = self.sub(f'prior{i}', nn.Linear, self.hidden, **self.kw)(x)
        x = nn.act(self.act)(self.sub(f'prior{i}norm', nn.Norm, self.norm)(x))
      return self._logit('priorlogit', x)
  def _logit(self, name, x):
    kw = dict(**self.kw, outscale=self.outscale)
    x = self.sub(name, nn.Linear, self.stoch * self.classes, **kw)(x)
    return x.reshape(x.shape[:-1] + (self.stoch, self.classes))

  def _spatial_logit(self, name, x):
    """Per-location logit: (..., HW, hidden) → (..., HW*stoch, classes)."""
    kw = dict(**self.kw, outscale=self.outscale)
    x = self.sub(name, nn.Linear, self.stoch * self.classes, **kw)(x)  # (..., HW, stoch*cls)
    *lead, HW, _ = x.shape
    return x.reshape(*lead, HW * self.stoch, self.classes)

  def _obs_logit_spatial(self, deter_prior, tokens):
    """Spatial posterior: per-location obslogit from (h_s_tok[i,j], enc_tok[i,j], h_v)."""
    H = W = self.lat_size; C = self.lat_chan; sp = self._sp()
    B = deter_prior.shape[0]
    h_s_tok = deter_prior[:, :sp].reshape(B, H * W, C)           # (B, HW, C)
    h_v = deter_prior[:, sp:]                                     # (B, D)
    C_enc = tokens.shape[-1] // (H * W)
    enc_tok = tokens.reshape(B, H * W, C_enc)                     # (B, HW, C_enc)
    x = enc_tok if self.absolute else jnp.concatenate(
        [h_s_tok, enc_tok], axis=-1)                              # (B, HW, C+C_enc)
    h_v_proj = self.sub('obs_v2s', nn.Linear, C, **self.kw)(h_v)  # (B, C)
    x = jnp.concatenate(
        [x, jnp.broadcast_to(h_v_proj[:, None, :], (B, H * W, C))],
        axis=-1)                                                   # (B, HW, *+C)
    for i in range(self.obslayers):
      x = self.sub(f'obs{i}', nn.Linear, self.hidden, **self.kw)(x)
      x = nn.act(self.act)(self.sub(f'obs{i}norm', nn.Norm, self.norm)(x))
    return self._spatial_logit('obslogit', x)                     # (B, HW*stoch, classes)



  def _spatial_op(self, h_s_tok, B, H, W, C, compute_dtype, act_emb):
    """Dispatch to the selected spatial operator. Returns (B, HW, C)."""
    if self.spatial_op == 'fno':
      return self._fno_op(h_s_tok, B, H, W, C, compute_dtype) + act_emb[:, None, :]
    elif self.spatial_op == 'attn':
      return self._attn_op(h_s_tok, B, H, W, C) + act_emb[:, None, :]
    elif self.spatial_op == 'conv':
      return self._conv_op(h_s_tok, B, H, W, C) + act_emb[:, None, :]
    elif self.spatial_op == 'wno':
      return self._wno_op(h_s_tok, B, H, W, C, act_emb)
    else:
      raise ValueError(f'Unknown spatial_op: {self.spatial_op!r}')

  def _fno_op(self, h_s_tok, B, H, W, C, compute_dtype):
    """FNO-2D: per-mode spectral channel mixing + per-location bypass.

    Fix vs naive impl: keep (B, m, m, 2C) shape so each Fourier mode gets its
    own channel mixing instead of a single flat transform over all modes at once.
    Parameters: Linear(2C, 2C) ≈ 1K vs flat Linear(m²·2C, m²·2C) ≈ 262K.
    """
    m = min(self.fno_modes, H // 2 + 1, W // 2 + 1)
    h_s = h_s_tok.reshape(B, H, W, C)
    h_fft = jnp.fft.rfft2(h_s.astype(jnp.float32), axes=(1, 2))
    h_fft_m = h_fft[:, :m, :m, :]                               # (B, m, m, C)
    h_ri = jnp.concatenate([h_fft_m.real, h_fft_m.imag], -1)   # (B, m, m, 2C)
    # Per-mode channel mixing (shared weight): proper FNO spectral operator
    h_ri = self.sub('fno_spec', nn.Linear, 2 * C, **self.kw)(
        h_ri.astype(compute_dtype))                              # (B, m, m, 2C)
    h_ri = h_ri.astype(jnp.float32)
    h_fft_new = jnp.zeros_like(h_fft).at[:, :m, :m, :].set(
        h_ri[..., :C] + 1j * h_ri[..., C:])
    h_spec = jnp.fft.irfft2(
        h_fft_new, s=(H, W), axes=(1, 2)).astype(compute_dtype)
    h_loc = self.sub('fno_loc', nn.Linear, C, **self.kw)(h_s_tok).reshape(B, H, W, C)
    return (h_spec + h_loc).reshape(B, H * W, C)

  def _wno_op(self, h_s_tok, B, H, W, C, act_emb):
    """Wavelet Neural Operator with action-conditioned per-subband FiLM.

    Haar DWT is compact-support (local) unlike Fourier, so it naturally
    represents localized objects (ball, paddle) at multiple scales.
    2-level decomposition on 8×8: level-1 subbands at 4×4, level-2 at 2×2.
    Per-subband: Linear(C,C) → FiLM(act_emb) → norm + act — 7 blocks.
    Action conditions the wavelet channel mixing directly (operator-level driving).
    Bypass path (like FNO h_loc) preserves pointwise features.
    """
    compute_dtype = h_s_tok.dtype
    h = h_s_tok.reshape(B, H, W, C)

    def haar_fwd(x):
      """Single-level 2D Haar forward in float32 to avoid bfloat16 cancellation."""
      x = x.astype(jnp.float32)
      a, b = x[:, ::2], x[:, 1::2]
      lo = (a + b) * 0.5
      hi = (a - b) * 0.5
      ll = (lo[:, :, ::2] + lo[:, :, 1::2]) * 0.5
      lh = (lo[:, :, ::2] - lo[:, :, 1::2]) * 0.5
      hl = (hi[:, :, ::2] + hi[:, :, 1::2]) * 0.5
      hh = (hi[:, :, ::2] - hi[:, :, 1::2]) * 0.5
      return (ll.astype(compute_dtype), lh.astype(compute_dtype),
              hl.astype(compute_dtype), hh.astype(compute_dtype))

    def haar_inv(ll, lh, hl, hh, h_out, w_out):
      """Single-level 2D Haar inverse via stack+reshape (exact reconstruction)."""
      # Inverse on cols: even cols = ll+lh, odd cols = ll-lh
      lo = jnp.stack([ll + lh, ll - lh], axis=3).reshape(B, h_out // 2, w_out, C)
      hi = jnp.stack([hl + hh, hl - hh], axis=3).reshape(B, h_out // 2, w_out, C)
      # Inverse on rows: even rows = lo+hi, odd rows = lo-hi
      return jnp.stack([lo + hi, lo - hi], axis=2).reshape(B, h_out, w_out, C)

    def mix(name, x):
      h = self.sub(name, nn.Linear, C, **self.kw)(x)
      film = self.sub(name + '_act', nn.Linear, 2 * C, **self.kw)(act_emb)  # (B, 2C)
      γ, β = jnp.split(film, 2, axis=-1)                                    # (B, C) each
      γ = jax.nn.sigmoid(γ) * 2                                             # ∈[0,2], ≈1 at init
      h = γ[:, None, None, :] * h + β[:, None, None, :]
      return nn.act(self.act)(self.sub(name + 'n', nn.Norm, self.norm)(h))

    # Level-1 decomposition
    ll1, lh1, hl1, hh1 = haar_fwd(h)
    # Level-2 decomposition on LL
    ll2, lh2, hl2, hh2 = haar_fwd(ll1)

    # Per-subband learnable channel mixing + norm + act
    ll2 = mix('wno_ll2', ll2)
    lh2 = mix('wno_lh2', lh2)
    hl2 = mix('wno_hl2', hl2)
    hh2 = mix('wno_hh2', hh2)
    lh1 = mix('wno_lh1', lh1)
    hl1 = mix('wno_hl1', hl1)
    hh1 = mix('wno_hh1', hh1)

    # Reconstruct level-2 → level-1 LL → full resolution
    ll1_rec = haar_inv(ll2, lh2, hl2, hh2, H // 2, W // 2)
    h_wno   = haar_inv(ll1_rec, lh1, hl1, hh1, H, W)

    # Bypass path: per-location linear (like FNO's h_loc), preserves pointwise features
    h_bypass = self.sub('wno_loc', nn.Linear, C, **self.kw)(h_s_tok).reshape(B, H, W, C)
    return (h_wno + h_bypass).reshape(B, H * W, C)

  def _attn_op(self, h_s_tok, B, H, W, C):
    """Multi-head self-attention over spatial tokens with pre-norm and residual.

    Pre-LN + residual matches standard transformer practice and stabilises
    training; without it the attention output can dominate h_s_tok.
    Complexity O(HW² × C) — cheap for 8×8 (64 tokens).
    """
    heads = self.attn_heads
    dh = max(1, C // heads)
    x = self.sub('attn_norm', nn.Norm, self.norm)(h_s_tok)         # pre-norm
    qkv = self.sub('attn_qkv', nn.Linear, 3 * heads * dh, **self.kw)(x)
    q, k, v = jnp.split(qkv, 3, axis=-1)                          # (B, HW, heads*dh)
    def split_heads(x):
      return x.reshape(B, H * W, heads, dh).transpose(0, 2, 1, 3)
    q, k, v = split_heads(q), split_heads(k), split_heads(v)       # (B, heads, HW, dh)
    attn = jax.nn.softmax(
        q @ k.transpose(0, 1, 3, 2) * (dh ** -0.5), axis=-1)
    out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, H * W, heads * dh)
    out = self.sub('attn_proj', nn.Linear, C, **self.kw)(out)
    return h_s_tok + out                                            # residual

  def _conv_op(self, h_s_tok, B, H, W, C):
    """3×3 local convolution via im2col + linear (zero-padded borders).

    Equivalent to a full (non-depthwise) 3×3 conv. Captures local motion
    patterns (ball/paddle movement between adjacent spatial cells).
    """
    h_s = h_s_tok.reshape(B, H, W, C)
    h_pad = jnp.pad(h_s, ((0, 0), (1, 1), (1, 1), (0, 0)))      # (B, H+2, W+2, C)
    patches = [h_pad[:, di:di + H, dj:dj + W, :]
               for di in range(3) for dj in range(3)]             # 9 × (B, H, W, C)
    h_patches = jnp.concatenate(patches, axis=-1)                 # (B, H, W, 9C)
    h_patches = h_patches.reshape(B, H * W, 9 * C)
    return self.sub('conv_proj', nn.Linear, C, **self.kw)(h_patches)  # (B, HW, C)

  def _dist(self, logits):
    out = embodied.jax.outs.OneHot(logits, self.unimix)
    out = embodied.jax.outs.Agg(out, 1, jnp.sum)
    return out
