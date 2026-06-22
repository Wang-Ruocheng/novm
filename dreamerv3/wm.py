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
  # NOWM spatial decode: set spatial_dec=True + lat_size/lat_chan to use
  # h_spatial as 2D starting feature map instead of flat BlockLinear projection.
  # Requires lat_size × 2^len(mults) == imgres (e.g. 8 × 2^3 = 64).
  # Use a dedicated bool so debug's .*\.lat_size regex doesn't accidentally
  # activate this path in non-NOWM runs.
  spatial_dec: bool = False
  lat_size: int = 0
  lat_chan: int = 0

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
    K = self.kernel
    recons = {}
    bshape = reset.shape
    BT = math.prod(bshape)
    inp = [nn.cast(feat[k]) for k in ('stoch', 'deter')]
    inp = [x.reshape((BT, -1)) for x in inp]
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
      if self.spatial_dec:
        # ── Spatial decode path (NOWM) ──────────────────────────────────────
        # Starting feature map: concat(h_spatial, stoch_spatial) at 8×8,
        # conditioned by h_vec via FiLM.  No flat projection, no resolution loss.
        H = W = self.lat_size; C_lat = self.lat_chan
        sp = H * W * C_lat
        assert self.lat_size * (2 ** len(self.depths)) == self.imgres[0], (
            f'lat_size={self.lat_size} × 2^{len(self.depths)} = '
            f'{self.lat_size * 2**len(self.depths)} ≠ imgres={self.imgres[0]}. '
            f'Adjust mults so len(mults) = log2(imgres/lat_size).')
        d = nn.cast(feat['deter']).reshape(BT, -1)
        h_s = d[:, :sp].reshape(BT, H, W, C_lat)          # (BT, H, W, lat_chan)
        h_v = d[:, sp:]                                    # (BT, deter)
        stoch_s = nn.cast(feat['stoch']).reshape(BT, H, W, -1)  # (BT, H, W, stoch*cls)
        x = jnp.concatenate([h_s, stoch_s], axis=-1)      # (BT, H, W, lat_chan+stoch*cls)
        x = self.sub('sp_proj', nn.Linear, self.depths[-1], **self.kw)(x)
        x = self.sub('sp_projn', nn.Norm, self.norm)(x)
        # h_vec → FiLM: global state modulates every spatial location
        film = self.sub('sp_film', nn.Linear, 2 * self.depths[-1], **self.kw)(h_v)
        γ, β = jnp.split(film, 2, axis=-1)
        γ = jax.nn.sigmoid(γ) * 2
        x = nn.act(self.act)(γ[:, None, None, :] * x + β[:, None, None, :])
      else:
        # ── Original bspace path (RSSM / non-spatial) ───────────────────────
        assert feat['deter'].shape[-1] % self.bspace == 0
        factor = 2 ** (len(self.depths) - int(bool(self.outer)))
        minres = [int(r // factor) for r in self.imgres]
        assert 3 <= minres[0] <= 16, minres
        assert 3 <= minres[1] <= 16, minres
        shape = (*minres, self.depths[-1])
        if self.bspace:
          u, g = math.prod(shape), self.bspace
          x0, x1 = nn.cast((feat['deter'], feat['stoch']))
          x0 = x0.reshape((BT, x0.shape[-1]))
          x1 = x1.reshape((BT, -1))
          x0 = self.sub('sp0', nn.BlockLinear, u, g, **self.kw)(x0)
          x0 = einops.rearrange(
              x0, '... (g h w c) -> ... h w (g c)',
              h=minres[0], w=minres[1], g=g)
          x1 = self.sub('sp1', nn.Linear, 2 * self.units, **self.kw)(x1)
          x1 = nn.act(self.act)(self.sub('sp1norm', nn.Norm, self.norm)(x1))
          x1 = self.sub('sp2', nn.Linear, shape, **self.kw)(x1)
          x = nn.act(self.act)(self.sub('spnorm', nn.Norm, self.norm)(x0 + x1))
        else:
          x = self.sub('space', nn.Linear, shape, **self.kw)(inp)
          x = nn.act(self.act)(self.sub('spacenorm', nn.Norm, self.norm)(x))
      # ── Shared upsampling ────────────────────────────────────────────────
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
    stoch_shape = (
        (self.lat_size, self.lat_size, self.stoch, self.classes)
        if self.stoch_spatial else
        (self._stoch_total(), self.classes))
    return dict(
        deter=elements.Space(np.float32, self._total()),
        stoch=elements.Space(np.float32, stoch_shape))

  def initial(self, bsize):
    if self.stoch_spatial:
      stoch = jnp.zeros(
          [bsize, self.lat_size, self.lat_size, self.stoch, self.classes], f32)
    else:
      stoch = jnp.zeros([bsize, self.stoch, self.classes], f32)
    return nn.cast(dict(
        deter=jnp.zeros([bsize, self._total()], f32),
        stoch=stoch,
        tokens_prev=jnp.zeros([bsize, self._sp()], f32)))

  def truncate(self, entries, carry=None):
    assert entries['deter'].ndim == 3, entries['deter'].shape
    result = jax.tree.map(lambda x: x[:, -1], entries)
    # tokens_prev is not stored in entries; restore from carry or default to zeros
    if carry is not None and 'tokens_prev' in carry:
      result['tokens_prev'] = carry['tokens_prev']
    else:
      B = result['deter'].shape[0]
      result['tokens_prev'] = jnp.zeros([B, self._sp()], result['deter'].dtype)
    return result

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
    deter, stoch, action, tokens_prev = nn.mask(
        (carry['deter'], carry['stoch'], action, carry['tokens_prev']), ~reset)
    action = nn.DictConcat(self.act_space, 1)(action)
    action = nn.mask(action, ~reset)
    deter = self._core(deter, stoch, action)
    tokens = tokens.reshape((*deter.shape[:-1], -1))

    # Velocity: project enc tokens to lat_chan dims, diff with previous step
    B = tokens.shape[0]; H = W = self.lat_size; C = self.lat_chan
    C_enc = tokens.shape[-1] // (H * W)
    tokens_proj = self.sub('vel_tok_proj', nn.Linear, C, **self.kw)(
        tokens.reshape(B, H * W, C_enc))                       # (B, HW, C)
    vel = tokens_proj - tokens_prev.reshape(B, H * W, C)       # (B, HW, C)

    deter_prior = deter
    if self.stoch_spatial:
      # Bottleneck: posterior computed from prior deter + enc_tok only.
      # h_spatial is then updated from stoch_post (not enc_tok directly),
      # forcing all observation info through the stochastic bottleneck.
      logit = self._obs_logit_spatial(deter_prior, tokens)
      stoch = nn.cast(self._dist(logit).sample(seed=nj.seed()))
      stoch = stoch.reshape(B, self.lat_size, self.lat_size, self.stoch, self.classes)
      deter = self._spatial_observe(deter_prior, tokens, vel, stoch_inject=stoch)
      deter = self._vec_observe(deter, tokens)
    else:
      deter = self._spatial_observe(deter_prior, tokens, vel)
      deter = self._vec_observe(deter, tokens)
      x = tokens if self.absolute else jnp.concatenate([deter, tokens], -1)
      for i in range(self.obslayers):
        x = self.sub(f'obs{i}', nn.Linear, self.hidden, **self.kw)(x)
        x = nn.act(self.act)(self.sub(f'obs{i}norm', nn.Norm, self.norm)(x))
      logit = self._logit('obslogit', x)
      stoch = nn.cast(self._dist(logit).sample(seed=nj.seed()))
    # carry/feat: deter_prior forces decoder to rely on stoch for obs info (no KL collapse).
    # entry:      deter_posterior for imagination starts (spatially-corrected state).
    carry = dict(deter=deter_prior, stoch=stoch,
                 tokens_prev=tokens_proj.reshape(B, -1))
    feat = dict(deter=deter_prior, prior_deter=deter_prior, stoch=stoch, logit=logit)
    entry = dict(deter=deter, stoch=stoch)
    assert all(x.dtype == nn.COMPUTE_DTYPE for x in (deter, stoch, logit))
    return carry, (entry, feat)

  def _spatial_observe(self, deter, tokens, vel, stoch_inject=None):
    """Spatial assimilation: h_spatial updated by obs-conditioned spatial operator.

    stoch_inject: (B, H, W, stoch, classes) — bottleneck mode.
      When provided, h_spatial is updated from stoch_post (not enc_tok directly),
      forcing all observation info to flow through the stochastic bottleneck.
      enc_tok is blocked from the injection path; only vel (motion delta) is kept.
      ctx uses h_v only (no enc_global), consistent with the bottleneck constraint.

    stoch_inject=None — direct mode (global stoch or non-spatial cases).
      enc_tok injected per-location; enc_global+h_v conditions the operator.

    In both modes, vel (per-location enc delta across timesteps) is injected as a
    motion signal — it encodes direction/speed rather than absolute appearance,
    so it does not create an enc_tok bypass.
    """
    H = W = self.lat_size
    C = self.lat_chan
    B = deter.shape[0]
    sp = self._sp()
    h_s_tok = deter[:, :sp].reshape(B, H * W, C)
    h_v = deter[:, sp:]

    assert tokens.shape[-1] % (H * W) == 0, (
        f'Token dim {tokens.shape[-1]} not divisible by H*W={H*W}.')
    C_enc = tokens.shape[-1] // (H * W)
    enc_tok = tokens.reshape(B, H * W, C_enc)     # (B, HW, C_enc)

    if stoch_inject is not None:
      # Bottleneck: inject stoch_post instead of enc_tok
      stoch_flat = stoch_inject.reshape(B, H * W, self.stoch * self.classes)
      h_s_in = h_s_tok + self.sub('assim_stoch_inj', nn.Linear, C, **self.kw)(stoch_flat)
      obs_ctx = nn.act(self.act)(
          self.sub('assim_ctx_norm', nn.Norm, self.norm)(
              self.sub('assim_ctx_s', nn.Linear, C, **self.kw)(h_v)))
    else:
      # Direct: inject enc_tok per-location
      if self.spatial_op != 'attnno':
        pe = self._pos2d(H, W, C).astype(h_s_tok.dtype)
        h_s_in = (h_s_tok + pe) + self.sub('assim_enc_inj', nn.Linear, C, **self.kw)(enc_tok) + pe
      else:
        h_s_in = h_s_tok + self.sub('assim_enc_inj', nn.Linear, C, **self.kw)(enc_tok)
      enc_global = enc_tok.mean(axis=1)
      obs_ctx = nn.act(self.act)(
          self.sub('assim_ctx_norm', nn.Norm, self.norm)(
              self.sub('assim_ctx', nn.Linear, C, **self.kw)(
                  jnp.concatenate([enc_global, h_v], axis=-1))))

    # Velocity injection: per-location change in projected enc tokens → direction/speed
    h_s_in = h_s_in + self.sub('assim_vel_inj', nn.Linear, C, **self.kw)(vel)
    h_s_in = self.sub('assim_in_norm', nn.Norm, self.norm)(h_s_in)

    h_s_cand = self._spatial_op(h_s_in, B, H, W, C, h_s_in.dtype, obs_ctx, prefix='assim')

    # Gated update (bias=-1 → near-identity at init)
    gate = jax.nn.sigmoid(
        self.sub('assim_gate', nn.Linear, C, **self.kw)(
            jnp.concatenate([h_s_tok, h_s_cand], axis=-1)) - 1)
    h_s_post = (gate * jnp.tanh(h_s_cand) + (1 - gate) * h_s_tok).reshape(B, sp)

    return jnp.concatenate([h_s_post, h_v], axis=-1)

  def _vec_observe(self, deter, tokens):
    """Global assimilation: h_vec updated via GRU conditioned on attended encoder tokens.

    Symmetric with _core's h_v GRU update:
      _core:  GRU(h_v, s2g + action + stoch)  — dynamics driven by spatial readout
      here:   GRU(h_v, enc_attended)           — assimilation driven by encoder readout

    h_vec queries encoder tokens via attention first (selects relevant spatial regions),
    then the attended summary drives a GRU update of h_v.
    """
    sp = self._sp()
    D = self.deter
    B = deter.shape[0]
    H = W = self.lat_size

    h_v = deter[:, sp:]

    C_enc = tokens.shape[-1] // (H * W)
    enc_tok = tokens.reshape(B, H * W, C_enc)   # (B, HW, C_enc)

    # h_vec-conditioned attention: select relevant spatial regions from encoder
    heads = self.attn_heads; dh = max(1, C_enc // heads)
    q_vo = self.sub('vobs_q', nn.Linear, heads * dh, **self.kw)(h_v).reshape(B, heads, dh)
    k_vo = (self.sub('vobs_k', nn.Linear, heads * dh, **self.kw)(enc_tok)
            .reshape(B, H * W, heads, dh).transpose(0, 2, 1, 3))
    v_vo = (self.sub('vobs_v', nn.Linear, heads * dh, **self.kw)(enc_tok)
            .reshape(B, H * W, heads, dh).transpose(0, 2, 1, 3))
    attn_vo = jax.nn.softmax(
        jnp.einsum('bhd,bhnd->bhn', q_vo, k_vo) * (dh ** -0.5), axis=-1)
    enc_attended = jnp.einsum('bhn,bhnd->bhd', attn_vo, v_vo).reshape(B, heads * dh)

    # GRU update (mirrors _core's h_v GRU, driven by enc_attended instead of s2g+action)
    gru_in = jnp.concatenate([h_v, enc_attended], axis=-1)
    gru_out = self.sub('vobs_gru', nn.Linear, 3 * D, **self.kw)(gru_in)
    r, cand_v, upd = jnp.split(gru_out, 3, axis=-1)
    r = jax.nn.sigmoid(r)
    cand_v = jnp.tanh(r * cand_v)
    upd = jax.nn.sigmoid(upd - 1)
    h_v_post = upd * cand_v + (1 - upd) * h_v                          # (B, D)

    return jnp.concatenate([deter[:, :sp], h_v_post], axis=-1)

  def imagine(self, carry, policy, length, training, single=False):
    carry = {k: carry[k] for k in ('deter', 'stoch')}  # strip velocity state
    if single:
      action = policy(sg(carry)) if callable(policy) else policy
      actemb = nn.DictConcat(self.act_space, 1)(action)
      deter = self._core(carry['deter'], carry['stoch'], actemb)
      logit = self._prior(deter)
      stoch = nn.cast(self._dist(logit).sample(seed=nj.seed()))
      if self.stoch_spatial:
        B = deter.shape[0]
        stoch = stoch.reshape(B, self.lat_size, self.lat_size, self.stoch, self.classes)
        # logit already (B, H, W, stoch, classes) from _spatial_logit — no reshape needed
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
    dyn_kl = self._kl_per_var(sg(post), prior)   # (..., N)
    rep_kl = self._kl_per_var(post, sg(prior))   # (..., N)
    dyn = dyn_kl.sum(-1)
    rep = rep_kl.sum(-1)
    if self.free_nats:
      dyn = jnp.maximum(dyn, self.free_nats)
      rep = jnp.maximum(rep, self.free_nats)
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
    stoch_flat = stoch.reshape(B, -1)
    action = action / sg(jnp.maximum(1, jnp.abs(action)))

    # Split deter → (h_spatial, h_vec)
    h_s = deter[:, :sp].reshape(B, H, W, C)   # (B, H, W, C)
    h_v = deter[:, sp:]                         # (B, D)
    h_s_tok = h_s.reshape(B, H * W, C)          # (B, HW, C) original — kept for identity path
    h_s_tok_norm = self.sub('mix_norm_s', nn.Norm, self.norm)(h_s_tok)  # normalized — for operator input

    # ---- Context embedding: action + global state ----
    # Both action and h_vec condition the spatial operator INSIDE WNO (per-subband FiLM).
    # Combining them here keeps the operator interface clean.
    act_emb = self.sub('op_act', nn.Linear, C, **self.kw)(action)      # (B, C)
    h_v_emb = self.sub('op_hv',  nn.Linear, C, **self.kw)(h_v)         # (B, C)
    ctx = nn.act(self.act)(
        self.sub('op_ctx_norm', nn.Norm, self.norm)(act_emb + h_v_emb))  # (B, C)

    # ---- Spatial operator: conditioned on ctx inside WNO subbands ----
    # For attnno, RoPE handles position internally — skip additive PE here.
    if self.spatial_op != 'attnno':
      h_s_tok_norm = h_s_tok_norm + self._pos2d(H, W, C).astype(h_s_tok_norm.dtype)
    h_s_mixed = self._spatial_op(h_s_tok_norm, B, H, W, C, deter.dtype, ctx)

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
    # Gate uses normalized scale (same as cand_s_tok); identity path uses raw h_s_tok.
    gate_s = jax.nn.sigmoid(
        self.sub('op_gate', nn.Linear, C, **self.kw)(
            jnp.concatenate([h_s_tok_norm, cand_s_tok], -1)) - 1)
    h_s_new = (
        gate_s * jnp.tanh(cand_s_tok) + (1 - gate_s) * h_s_tok
    ).reshape(B, H, W, C)

    # ---- Spatial → global: h_vec-conditioned attention pooling ----
    # h_vec acts as query: it actively selects which spatial locations are relevant
    # (e.g. "health low" → attend to nearby hazards), rather than blind mean+max.
    h_s_tok_new = h_s_new.reshape(B, H * W, C)
    heads = self.attn_heads; dh = max(1, C // heads)
    q_s2g = self.sub('s2g_q', nn.Linear, heads * dh, **self.kw)(h_v).reshape(B, heads, dh)
    k_s2g = (self.sub('s2g_k', nn.Linear, heads * dh, **self.kw)(h_s_tok_new)
             .reshape(B, H * W, heads, dh).transpose(0, 2, 1, 3))
    v_s2g = (self.sub('s2g_v', nn.Linear, heads * dh, **self.kw)(h_s_tok_new)
             .reshape(B, H * W, heads, dh).transpose(0, 2, 1, 3))
    attn_s2g = jax.nn.softmax(
        jnp.einsum('bhd,bhnd->bhn', q_s2g, k_s2g) * (dh ** -0.5), axis=-1)
    s2g = jnp.einsum('bhn,bhnd->bhd', attn_s2g, v_s2g).reshape(B, heads * dh)
    s2g = self.sub('s2g_proj', nn.Linear, self.hidden, **self.kw)(s2g)
    s2g = nn.act(self.act)(self.sub('s2g_norm', nn.Norm, self.norm)(s2g))

    # ---- GRU update for h_vec ----
    # Spatial stoch is (B, H*W*stoch*classes): project to hidden before concat
    if self.stoch_spatial:
      stoch_gru = self.sub('stoch_enc', nn.Linear, self.hidden, **self.kw)(stoch_flat)
      stoch_gru = nn.act(self.act)(self.sub('stoch_enc_norm', nn.Norm, self.norm)(stoch_gru))
    else:
      stoch_gru = stoch_flat
    gru_in = jnp.concatenate([h_v, s2g, action, stoch_gru], -1)
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
      h_s_tok = feat[..., :sp].reshape((*feat.shape[:-1], H * W, C))  # (..., HW, C)
      h_v = feat[..., sp:]                                           # (..., D)
      x = h_s_tok
      for i in range(self.imglayers):
        x = self.sub(f'prior{i}', nn.Linear, self.hidden, **self.kw)(x)   # (..., HW, hidden)
        x = self.sub(f'prior{i}norm', nn.Norm, self.norm)(x)
        ctx_p = nn.act(self.act)(
            self.sub(f'prior{i}_ctx_norm', nn.Norm, self.norm)(
                self.sub(f'prior{i}_ctx', nn.Linear, self.hidden, **self.kw)(h_v)))
        film = self.sub(f'prior{i}_film', nn.Linear, 2 * self.hidden, **self.kw)(ctx_p)
        γ, β = jnp.split(film, 2, axis=-1)
        γ = jax.nn.sigmoid(γ) * 2
        x = nn.act(self.act)(γ[..., None, :] * x + β[..., None, :])
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
    """Per-location logit: (..., HW, hidden) → (..., H, W, stoch, classes)."""
    H = W = self.lat_size
    kw = dict(**self.kw, outscale=self.outscale)
    x = self.sub(name, nn.Linear, self.stoch * self.classes, **kw)(x)  # (..., HW, stoch*cls)
    *lead, HW, _ = x.shape
    return x.reshape((*lead, H, W, self.stoch, self.classes))

  def _obs_logit_spatial(self, deter, tokens):
    """Spatial posterior: per-location obslogit from prior deter + encoder tokens.

    Receives deter_prior (before assimilation) so the posterior is computed from
    the dynamics prediction + observation, without enc_tok contamination in deter.
    KL/reconstruction gradients flow back through tokens and deter_prior → _core.
"""
    H = W = self.lat_size; C = self.lat_chan; sp = self._sp()
    B = deter.shape[0]
    h_s_tok = deter[:, :sp].reshape(B, H * W, C)               # (B, HW, C)
    h_v = deter[:, sp:]                                         # (B, D)
    C_enc = tokens.shape[-1] // (H * W)
    enc_tok = tokens.reshape(B, H * W, C_enc)                   # (B, HW, C_enc)
    # Pre-norm: align latent and encoder feature scales before concat
    # (mirrors _spatial_observe's assim_in_norm; h_s_tok is in GRU-output space,
    #  enc_tok is in CNN-activation space — their scales differ by initialization)
    h_s_tok = self.sub('obs_hs_norm', nn.Norm, self.norm)(h_s_tok)
    enc_tok = self.sub('obs_enc_norm', nn.Norm, self.norm)(enc_tok)
    x = enc_tok if self.absolute else jnp.concatenate(
        [h_s_tok, enc_tok], axis=-1)                            # (B, HW, C+C_enc)
    h_v_proj = self.sub('obs_v2s', nn.Linear, C, **self.kw)(h_v)  # (B, C)
    x = jnp.concatenate(
        [x, jnp.broadcast_to(h_v_proj[:, None, :], (B, H * W, C))],
        axis=-1)                                                 # (B, HW, *+C)
    for i in range(self.obslayers):
      x = self.sub(f'obs{i}', nn.Linear, self.hidden, **self.kw)(x)
      x = nn.act(self.act)(self.sub(f'obs{i}norm', nn.Norm, self.norm)(x))
    return self._spatial_logit('obslogit', x)                   # (B, H, W, stoch, classes)


  def _spatial_op(self, h_s_tok, B, H, W, C, compute_dtype, ctx, prefix='dyn'):
    """Dispatch to the selected spatial operator. Returns (B, HW, C).

    prefix distinguishes dynamics weights ('dyn') from assimilation weights ('assim'),
    allowing the same operator structure to be reused with independent parameters.
    fno/attn/conv use FiLM conditioning on ctx; wno handles its own per-subband FiLM.
    """
    if self.spatial_op == 'fno':
      out = self._fno_op(h_s_tok, B, H, W, C, compute_dtype, prefix)
    elif self.spatial_op == 'attn':
      out = self._attn_op(h_s_tok, B, H, W, C, prefix)
    elif self.spatial_op == 'conv':
      out = self._conv_op(h_s_tok, B, H, W, C, prefix)
    elif self.spatial_op == 'wno':
      return self._wno_op(h_s_tok, B, H, W, C, ctx, prefix)
    elif self.spatial_op == 'attnno':
      return self._attn_no_op(h_s_tok, B, H, W, C, ctx, prefix)
    else:
      raise ValueError(f'Unknown spatial_op: {self.spatial_op!r}')
    film = self.sub(f'{prefix}_ctx_film', nn.Linear, 2 * C, **self.kw)(ctx)
    γ, β = jnp.split(film, 2, axis=-1)
    γ = jax.nn.sigmoid(γ) * 2
    return γ[:, None, :] * out + β[:, None, :]

  def _fno_op(self, h_s_tok, B, H, W, C, compute_dtype, prefix='dyn'):
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
    h_ri = self.sub(f'{prefix}_spec', nn.Linear, 2 * C, **self.kw)(
        h_ri.astype(compute_dtype))                              # (B, m, m, 2C)
    h_ri = h_ri.astype(jnp.float32)
    h_fft_new = jnp.pad(
        (h_ri[..., :C] + 1j * h_ri[..., C:]).astype(jnp.complex64),
        ((0, 0), (0, H - m), (0, W // 2 + 1 - m), (0, 0)))
    h_spec = jnp.fft.irfft2(
        h_fft_new, s=(H, W), axes=(1, 2)).astype(compute_dtype)
    h_loc = self.sub(f'{prefix}_loc', nn.Linear, C, **self.kw)(h_s_tok).reshape(B, H, W, C)
    return (h_spec + h_loc).reshape(B, H * W, C)

  def _wno_op(self, h_s_tok, B, H, W, C, ctx, prefix='dyn'):
    """Wavelet Neural Operator conditioned on context (action + global state).

    ctx = f(action, h_vec) conditions each subband's channel mixing via FiLM,
    so the operator's behavior adapts to both what action is taken AND the
    current global state (score, health, etc.).
    """
    compute_dtype = h_s_tok.dtype
    assert H % 4 == 0 and W % 4 == 0, (
        f'_wno_op requires lat_size divisible by 4, got {H}×{W}')
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
      h = self.sub(f'{prefix}_{name}', nn.Linear, C, **self.kw)(x)
      h = self.sub(f'{prefix}_{name}n', nn.Norm, self.norm)(h)          # norm first
      film = self.sub(f'{prefix}_{name}_act', nn.Linear, 2 * C, **self.kw)(ctx)  # (B, 2C)
      γ, β = jnp.split(film, 2, axis=-1)
      γ = jax.nn.sigmoid(γ) * 2
      return nn.act(self.act)(γ[:, None, None, :] * h + β[:, None, None, :])

    # Level-1 decomposition
    ll1, lh1, hl1, hh1 = haar_fwd(h)
    # Level-2 decomposition on LL
    ll2, lh2, hl2, hh2 = haar_fwd(ll1)

    # Per-subband learnable channel mixing + norm + act
    ll2 = mix('ll2', ll2)
    lh2 = mix('lh2', lh2)
    hl2 = mix('hl2', hl2)
    hh2 = mix('hh2', hh2)
    lh1 = mix('lh1', lh1)
    hl1 = mix('hl1', hl1)
    hh1 = mix('hh1', hh1)

    # Reconstruct level-2 → level-1 LL → full resolution
    ll1_rec = haar_inv(ll2, lh2, hl2, hh2, H // 2, W // 2)
    h_wno   = haar_inv(ll1_rec, lh1, hl1, hh1, H, W)

    # Bypass path: per-location linear (like FNO's h_loc), preserves pointwise features
    h_bypass = self.sub(f'{prefix}_loc', nn.Linear, C, **self.kw)(h_s_tok).reshape(B, H, W, C)
    return (h_wno + h_bypass).reshape(B, H * W, C)

  def _attn_op(self, h_s_tok, B, H, W, C, prefix='dyn'):
    """Multi-head self-attention over spatial tokens with pre-norm and residual.

    Pre-LN + residual matches standard transformer practice and stabilises
    training; without it the attention output can dominate h_s_tok.
    Complexity O(HW² × C) — cheap for 8×8 (64 tokens).
    """
    heads = self.attn_heads
    dh = max(1, C // heads)
    x = self.sub(f'{prefix}_norm', nn.Norm, self.norm)(h_s_tok)         # pre-norm
    qkv = self.sub(f'{prefix}_qkv', nn.Linear, 3 * heads * dh, **self.kw)(x)
    q, k, v = jnp.split(qkv, 3, axis=-1)                          # (B, HW, heads*dh)
    def split_heads(x):
      return x.reshape(B, H * W, heads, dh).transpose(0, 2, 1, 3)
    q, k, v = split_heads(q), split_heads(k), split_heads(v)       # (B, heads, HW, dh)
    attn = jax.nn.softmax(
        q @ k.transpose(0, 1, 3, 2) * (dh ** -0.5), axis=-1)
    out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, H * W, heads * dh)
    out = self.sub(f'{prefix}_proj', nn.Linear, C, **self.kw)(out)
    return h_s_tok + out                                            # residual

  def _rope2d(self, q, k, H, W):
    """2D Rotary Position Embedding applied to Q and K.

    Splits head_dim into two halves: first half encodes row, second half col.
    Non-interleaved (GPT-NeoX) convention: rotate_half([a,b]) = [−b, a].
    The dot product qᵢ·kⱼ then depends only on (row_i−row_j, col_i−col_j),
    not absolute coordinates — better suited for translation-invariant dynamics.
    Falls back silently when dh < 4 or dh % 4 ≠ 0 (e.g. debug configs).
    """
    dh = q.shape[-1]
    if dh < 4 or dh % 4 != 0:
      return q, k
    d_half = dh // 2        # dims per spatial direction
    d_pair = d_half // 2    # independent frequency bands per direction
    freq = 1.0 / (10000.0 ** (jnp.arange(d_pair, dtype=jnp.float32) /
                               max(d_pair - 1, 1)))
    theta_row = jnp.outer(jnp.arange(H, dtype=jnp.float32), freq)   # (H, d_pair)
    theta_col = jnp.outer(jnp.arange(W, dtype=jnp.float32), freq)   # (W, d_pair)
    theta_row = jnp.broadcast_to(theta_row[:, None, :], (H, W, d_pair)).reshape(H * W, d_pair)
    theta_col = jnp.broadcast_to(theta_col[None, :, :], (H, W, d_pair)).reshape(H * W, d_pair)
    cos_r = jnp.tile(jnp.cos(theta_row), 2)    # (N, d_half) — non-interleaved
    sin_r = jnp.tile(jnp.sin(theta_row), 2)
    cos_c = jnp.tile(jnp.cos(theta_col), 2)
    sin_c = jnp.tile(jnp.sin(theta_col), 2)
    cos = jnp.concatenate([cos_r, cos_c], -1)   # (N, dh) float32
    sin = jnp.concatenate([sin_r, sin_c], -1)

    def rotate_half(x):
      xr, xc = x[..., :d_half], x[..., d_half:]
      def rh(t):
        t1, t2 = t[..., :d_pair], t[..., d_pair:]
        return jnp.concatenate([-t2, t1], -1)
      return jnp.concatenate([rh(xr), rh(xc)], -1)

    cos4 = cos[None, None].astype(jnp.float32)   # (1, 1, N, dh)
    sin4 = sin[None, None].astype(jnp.float32)
    q_f, k_f = q.astype(jnp.float32), k.astype(jnp.float32)
    return q_f * cos4 + rotate_half(q_f) * sin4, k_f * cos4 + rotate_half(k_f) * sin4

  def _attn_no_op(self, h_s_tok, B, H, W, C, ctx, prefix='dyn'):
    """Transformer NO: 2D RoPE + QK-Norm + ctx-token conditioning.

    Three improvements over plain _attn_op:
      1. QK-Norm (RMSNorm on Q/K after projection): bounds attention logit growth,
         prevents entropy collapse, stabilises early training.
      2. 2D RoPE on Q/K: encodes relative spatial position (Δrow, Δcol) via
         rotation — better than additive PE for translation-invariant dynamics.
      3. ctx as extra K/V token: action+state is a full token all spatial positions
         can attend to, more expressive than a scalar logit bias.
    Handles ctx internally; _spatial_op must NOT apply a second FiLM.
    """
    heads = self.attn_heads
    dh = max(1, C // heads)
    N = H * W

    # Pre-norm; no additive PE — RoPE handles position inside attention
    x = self.sub(f'{prefix}_no_norm', nn.Norm, self.norm)(h_s_tok)   # (B, N, C)

    q = self.sub(f'{prefix}_no_q', nn.Linear, heads * dh, **self.kw)(x)
    k = self.sub(f'{prefix}_no_k', nn.Linear, heads * dh, **self.kw)(x)
    v = self.sub(f'{prefix}_no_v', nn.Linear, heads * dh, **self.kw)(x)

    def split_heads(t):
      return t.reshape(B, N, heads, dh).transpose(0, 2, 1, 3)   # (B, heads, N, dh)
    q, k, v = split_heads(q), split_heads(k), split_heads(v)

    # QK-Norm per head (after split): normalise over dh so each head has unit RMS,
    # matching the dh**-0.5 scaling in logits.  Must come after split_heads.
    q = self.sub(f'{prefix}_no_qnorm', nn.Norm, self.norm)(
        q.reshape(B * heads, N, dh)).reshape(B, heads, N, dh)
    k = self.sub(f'{prefix}_no_knorm', nn.Norm, self.norm)(
        k.reshape(B * heads, N, dh)).reshape(B, heads, N, dh)

    # 2D RoPE encodes relative position into Q/K
    q, k = self._rope2d(q, k, H, W)
    q, k = q.astype(h_s_tok.dtype), k.astype(h_s_tok.dtype)

    # ctx token: all spatial positions can attend to the action/state context
    ctx_k = self.sub(f'{prefix}_no_ctx_k', nn.Linear, heads * dh, **self.kw)(ctx)
    ctx_v = self.sub(f'{prefix}_no_ctx_v', nn.Linear, heads * dh, **self.kw)(ctx)
    # QK-Norm on ctx_k so it stays on the same scale as the normalised spatial keys
    ctx_k = self.sub(f'{prefix}_no_ctx_knorm', nn.Norm, self.norm)(
        ctx_k.reshape(B, heads, dh)).reshape(B, heads, dh)
    ctx_k = ctx_k.reshape(B, 1, heads, dh).transpose(0, 2, 1, 3)   # (B, heads, 1, dh)
    ctx_v = ctx_v.reshape(B, 1, heads, dh).transpose(0, 2, 1, 3)
    k_full = jnp.concatenate([k, ctx_k], axis=2)    # (B, heads, N+1, dh)
    v_full = jnp.concatenate([v, ctx_v], axis=2)

    logits = jnp.einsum('bhid,bhjd->bhij', q, k_full).astype(jnp.float32) * (dh ** -0.5)
    attn = jax.nn.softmax(logits, axis=-1).astype(h_s_tok.dtype)    # (B, heads, N, N+1)
    out = jnp.einsum('bhij,bhjd->bhid', attn, v_full)               # (B, heads, N, dh)
    out = out.transpose(0, 2, 1, 3).reshape(B, N, heads * dh)
    out = self.sub(f'{prefix}_no_proj', nn.Linear, C, **self.kw)(out)

    out = h_s_tok + out   # residual

    # Lightweight output FiLM catches residual ctx signal
    film = self.sub(f'{prefix}_no_film', nn.Linear, 2 * C, **self.kw)(ctx)
    γ, β = jnp.split(film, 2, axis=-1)
    γ = jax.nn.sigmoid(γ) * 2
    return γ[:, None, :] * out + β[:, None, :]

  def _conv_op(self, h_s_tok, B, H, W, C, prefix='dyn'):
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
    return self.sub(f'{prefix}_proj', nn.Linear, C, **self.kw)(h_patches)  # (B, HW, C)

  def _dist(self, logits):
    if logits.ndim >= 5:  # spatial: (..., H, W, stoch, classes)
      *lead, H, W, S, C = logits.shape
      logits = logits.reshape((*lead, H * W * S, C))
    out = embodied.jax.outs.OneHot(logits, self.unimix)
    out = embodied.jax.outs.Agg(out, 1, jnp.sum)
    return out

  def _pos2d(self, H, W, C):
    """Fixed 2D sinusoidal position embedding, shape (H*W, C).

    Splits C evenly into 4 groups: sin(row), cos(row), sin(col), cos(col).
    No learnable parameters — acts as a stable positional prior so the spatial
    operator and assimilation cross-attention don't need to discover alignment
    from data.  Added to h_s_tok before WNO and before assimilation.
    """
    assert C % 4 == 0, f'lat_chan={C} must be divisible by 4 for 2D sinusoidal PE'
    d = C // 4
    freq = 1.0 / (10000 ** (jnp.arange(d, dtype=jnp.float32) / max(d - 1, 1)))
    rows = jnp.arange(H, dtype=jnp.float32)
    cols = jnp.arange(W, dtype=jnp.float32)
    row_enc = jnp.outer(rows, freq)                                  # (H, d)
    col_enc = jnp.outer(cols, freq)                                  # (W, d)
    row_pe = jnp.concatenate([jnp.sin(row_enc), jnp.cos(row_enc)], -1)  # (H, 2d)
    col_pe = jnp.concatenate([jnp.sin(col_enc), jnp.cos(col_enc)], -1)  # (W, 2d)
    row_pe = jnp.broadcast_to(row_pe[:, None, :], (H, W, 2 * d))
    col_pe = jnp.broadcast_to(col_pe[None, :, :], (H, W, 2 * d))
    pe = jnp.concatenate([row_pe, col_pe], -1)                      # (H, W, C)
    return pe.reshape(H * W, C)                                      # (HW, C)

  def _kl_per_var(self, post_logits, prior_logits):
    """Per-variable KL (..., N) before summing — for correct per-location free_nats floor."""
    def _prep(logits):
      if logits.ndim >= 5:
        *lead, H, W, S, C = logits.shape
        logits = logits.reshape((*lead, H * W * S, C))
      return embodied.jax.outs.OneHot(logits, self.unimix)
    return _prep(post_logits).kl(_prep(prior_logits))
