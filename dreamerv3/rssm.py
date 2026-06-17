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


class RSSM(nj.Module):

  deter: int = 4096
  hidden: int = 2048
  stoch: int = 32
  classes: int = 32
  norm: str = 'rms'
  act: str = 'gelu'
  unroll: bool = False
  unimix: float = 0.01
  outscale: float = 1.0
  imglayers: int = 2
  obslayers: int = 1
  dynlayers: int = 1
  absolute: bool = False
  blocks: int = 8
  free_nats: float = 1.0
  # Dynamics core: 'gru' = original block-GRU, 'fno' = neural operator.
  core: str = 'gru'
  fno_modes: int = 16

  def __init__(self, act_space, **kw):
    assert self.deter % self.blocks == 0
    self.act_space = act_space
    self.kw = kw

  @property
  def entry_space(self):
    return dict(
        deter=elements.Space(np.float32, self.deter),
        stoch=elements.Space(np.float32, (self.stoch, self.classes)))

  def initial(self, bsize):
    carry = nn.cast(dict(
        deter=jnp.zeros([bsize, self.deter], f32),
        stoch=jnp.zeros([bsize, self.stoch, self.classes], f32)))
    return carry

  def truncate(self, entries, carry=None):
    assert entries['deter'].ndim == 3, entries['deter'].shape
    carry = jax.tree.map(lambda x: x[:, -1], entries)
    return carry

  def starts(self, entries, carry, nlast):
    B = len(jax.tree.leaves(carry)[0])
    return jax.tree.map(
        lambda x: x[:, -nlast:].reshape((B * nlast, *x.shape[2:])), entries)

  def observe(self, carry, tokens, action, reset, training, single=False):
    carry, tokens, action = nn.cast((carry, tokens, action))
    if single:
      carry, (entry, feat) = self._observe(
          carry, tokens, action, reset, training)
      return carry, entry, feat
    else:
      unroll = jax.tree.leaves(tokens)[0].shape[1] if self.unroll else 1
      carry, (entries, feat) = nj.scan(
          lambda carry, inputs: self._observe(
              carry, *inputs, training),
          carry, (tokens, action, reset), unroll=unroll, axis=1)
      return carry, entries, feat

  def _observe(self, carry, tokens, action, reset, training):
    deter, stoch, action = nn.mask(
        (carry['deter'], carry['stoch'], action), ~reset)
    action = nn.DictConcat(self.act_space, 1)(action)
    action = nn.mask(action, ~reset)
    deter = self._core(deter, stoch, action)
    tokens = tokens.reshape((*deter.shape[:-1], -1))
    x = tokens if self.absolute else jnp.concatenate([deter, tokens], -1)
    for i in range(self.obslayers):
      x = self.sub(f'obs{i}', nn.Linear, self.hidden, **self.kw)(x)
      x = nn.act(self.act)(self.sub(f'obs{i}norm', nn.Norm, self.norm)(x))
    logit = self._logit('obslogit', x)
    stoch = nn.cast(self._dist(logit).sample(seed=nj.seed()))
    carry = dict(deter=deter, stoch=stoch)
    feat = dict(deter=deter, stoch=stoch, logit=logit)
    entry = dict(deter=deter, stoch=stoch)
    assert all(x.dtype == nn.COMPUTE_DTYPE for x in (deter, stoch, logit))
    return carry, (entry, feat)

  def imagine(self, carry, policy, length, training, single=False):
    if single:
      action = policy(sg(carry)) if callable(policy) else policy
      actemb = nn.DictConcat(self.act_space, 1)(action)
      deter = self._core(carry['deter'], carry['stoch'], actemb)
      logit = self._prior(deter)
      stoch = nn.cast(self._dist(logit).sample(seed=nj.seed()))
      carry = nn.cast(dict(deter=deter, stoch=stoch))
      feat = nn.cast(dict(deter=deter, stoch=stoch, logit=logit))
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
      # We can also return all carry entries but it might be expensive.
      # entries = dict(deter=feat['deter'], stoch=feat['stoch'])
      # return carry, entries, feat, action
      return carry, feat, action

  def loss(self, carry, tokens, acts, reset, training):
    metrics = {}
    carry, entries, feat = self.observe(carry, tokens, acts, reset, training)
    prior = self._prior(feat['deter'])
    post = feat['logit']
    dyn = self._dist(sg(post)).kl(self._dist(prior))
    rep = self._dist(post).kl(self._dist(sg(prior)))
    if self.free_nats:
      dyn = jnp.maximum(dyn, self.free_nats)
      rep = jnp.maximum(rep, self.free_nats)
    losses = {'dyn': dyn, 'rep': rep}
    metrics['dyn_ent'] = self._dist(prior).entropy().mean()
    metrics['rep_ent'] = self._dist(post).entropy().mean()
    return carry, entries, losses, feat, metrics

  def _core(self, deter, stoch, action):
    if self.core == 'fno':
      return self._core_fno(deter, stoch, action)
    return self._core_gru(deter, stoch, action)

  def _core_gru(self, deter, stoch, action):
    stoch = stoch.reshape((stoch.shape[0], -1))
    action /= sg(jnp.maximum(1, jnp.abs(action)))
    g = self.blocks
    flat2group = lambda x: einops.rearrange(x, '... (g h) -> ... g h', g=g)
    group2flat = lambda x: einops.rearrange(x, '... g h -> ... (g h)', g=g)
    x0 = self.sub('dynin0', nn.Linear, self.hidden, **self.kw)(deter)
    x0 = nn.act(self.act)(self.sub('dynin0norm', nn.Norm, self.norm)(x0))
    x1 = self.sub('dynin1', nn.Linear, self.hidden, **self.kw)(stoch)
    x1 = nn.act(self.act)(self.sub('dynin1norm', nn.Norm, self.norm)(x1))
    x2 = self.sub('dynin2', nn.Linear, self.hidden, **self.kw)(action)
    x2 = nn.act(self.act)(self.sub('dynin2norm', nn.Norm, self.norm)(x2))
    x = jnp.concatenate([x0, x1, x2], -1)[..., None, :].repeat(g, -2)
    x = group2flat(jnp.concatenate([flat2group(deter), x], -1))
    for i in range(self.dynlayers):
      x = self.sub(f'dynhid{i}', nn.BlockLinear, self.deter, g, **self.kw)(x)
      x = nn.act(self.act)(self.sub(f'dynhid{i}norm', nn.Norm, self.norm)(x))
    x = self.sub('dyngru', nn.BlockLinear, 3 * self.deter, g, **self.kw)(x)
    gates = jnp.split(flat2group(x), 3, -1)
    reset, cand, update = [group2flat(x) for x in gates]
    reset = jax.nn.sigmoid(reset)
    cand = jnp.tanh(reset * cand)
    update = jax.nn.sigmoid(update - 1)
    deter = update * cand + (1 - update) * deter
    return deter

  def _core_fno(self, deter, stoch, action):
    # PDE analogy: u_{t+1} = u_t + L[u_t] + f(a_t) + H(z_t)
    #   L[u]   = spectral (FNO) operator on hidden state (autonomous dynamics)
    #   f(a_t) = action forcing term
    #   H(z_t) = stochastic term via posterior sample
    stoch = stoch.reshape((stoch.shape[0], -1))
    action /= sg(jnp.maximum(1, jnp.abs(action)))
    B = deter.shape[0]
    g = self.blocks
    d = self.deter // g                          # spatial points per block
    m = min(self.fno_modes, d // 2 + 1)         # kept Fourier modes

    # --- spectral operator L[u]: FNO on deter reshaped to (B, g, d) ---
    # rfft requires float32; cast up, operate, cast back to compute dtype
    compute_dtype = deter.dtype
    h = deter.reshape(B, g, d).astype(jnp.float32)
    h_fft = jnp.fft.rfft(h, axis=-1)            # (B, g, d//2+1) complex64
    h_fft_m = h_fft[..., :m]                    # (B, g, m) keep low-freq modes
    # Represent complex modes as stacked real/imag, mix with a linear layer
    h_ri = jnp.concatenate(
        [h_fft_m.real, h_fft_m.imag], axis=-1)  # (B, g, 2m) float32
    h_ri = h_ri.reshape(B, g * 2 * m).astype(compute_dtype)
    h_ri = self.sub('fno_spec', nn.Linear, g * 2 * m, **self.kw)(h_ri)
    h_ri = h_ri.reshape(B, g, 2 * m).astype(jnp.float32)
    # Reconstruct complex spectrum and pad high frequencies with zero
    h_fft_new = jnp.zeros_like(h_fft).at[..., :m].set(
        h_ri[..., :m] + 1j * h_ri[..., m:])
    h_spectral = jnp.fft.irfft(h_fft_new, n=d, axis=-1)  # (B, g, d) float32
    h_spectral = h_spectral.reshape(B, self.deter).astype(compute_dtype)

    # Bypass (local W path): standard block-linear in spatial domain
    h_local = self.sub(
        'fno_loc', nn.BlockLinear, self.deter, g, **self.kw)(deter)

    # Merge spectral + local → autonomous candidate
    cand = h_spectral + h_local
    cand = nn.act(self.act)(self.sub('fno_candnorm', nn.Norm, self.norm)(cand))

    # --- forcing term f(a_t) ---
    forcing = self.sub('fno_force', nn.Linear, self.deter, **self.kw)(action)
    forcing = nn.act(self.act)(
        self.sub('fno_forcenorm', nn.Norm, self.norm)(forcing))

    # --- stochastic term H(z_t) ---
    noise = self.sub('fno_noise', nn.Linear, self.deter, **self.kw)(stoch)
    noise = nn.act(self.act)(
        self.sub('fno_noisenorm', nn.Norm, self.norm)(noise))

    # Combined: L[u] + f(a) + H(z)
    cand = cand + forcing + noise

    # --- gated update (GRU-style, initialised near identity for stability) ---
    # update gate biased to -1 so training starts close to identity map
    update = jax.nn.sigmoid(
        self.sub('fno_update', nn.Linear, self.deter, **self.kw)(
            jnp.concatenate([deter, cand], -1)) - 1)
    deter = update * jnp.tanh(cand) + (1 - update) * deter
    return deter

  def _prior(self, feat):
    x = feat
    for i in range(self.imglayers):
      x = self.sub(f'prior{i}', nn.Linear, self.hidden, **self.kw)(x)
      x = nn.act(self.act)(self.sub(f'prior{i}norm', nn.Norm, self.norm)(x))
    return self._logit('priorlogit', x)

  def _logit(self, name, x):
    kw = dict(**self.kw, outscale=self.outscale)
    x = self.sub(name, nn.Linear, self.stoch * self.classes, **kw)(x)
    return x.reshape(x.shape[:-1] + (self.stoch, self.classes))

  def _dist(self, logits):
    out = embodied.jax.outs.OneHot(logits, self.unimix)
    out = embodied.jax.outs.Agg(out, 1, jnp.sum)
    return out


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
  h_spatial (B, lat_size, lat_size, lat_chan): FNO-2D PDE dynamics
  h_vec     (B, deter): GRU global state
  Cross-attention (4 heads) connects the two streams each step.
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
  # FNO-2D: kept Fourier modes per spatial dimension
  fno_modes: int = 4

  def __init__(self, act_space, **kw):
    self.act_space = act_space
    self.kw = kw

  def _sp(self):
    return self.lat_size * self.lat_size * self.lat_chan

  def _total(self):
    return self._sp() + self.deter

  @property
  def entry_space(self):
    return dict(
        deter=elements.Space(np.float32, self._total()),
        stoch=elements.Space(np.float32, (self.stoch, self.classes)))

  def initial(self, bsize):
    return nn.cast(dict(
        deter=jnp.zeros([bsize, self._total()], f32),
        stoch=jnp.zeros([bsize, self.stoch, self.classes], f32)))

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
    x = tokens if self.absolute else jnp.concatenate([deter, tokens], -1)
    for i in range(self.obslayers):
      x = self.sub(f'obs{i}', nn.Linear, self.hidden, **self.kw)(x)
      x = nn.act(self.act)(self.sub(f'obs{i}norm', nn.Norm, self.norm)(x))
    logit = self._logit('obslogit', x)
    stoch = nn.cast(self._dist(logit).sample(seed=nj.seed()))
    carry = dict(deter=deter, stoch=stoch)
    feat = dict(deter=deter, prior_deter=deter_prior, stoch=stoch, logit=logit)
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
    enc_global = enc_tok.mean(axis=1)            # (B, C_enc) — global average pool

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
    compute_dtype = deter.dtype
    stoch_flat = stoch.reshape(B, -1)
    action = action / sg(jnp.maximum(1, jnp.abs(action)))

    # Split deter → (h_spatial, h_vec)
    h_s = deter[:, :sp].reshape(B, H, W, C)   # (B, H, W, C)
    h_v = deter[:, sp:]                         # (B, D)
    h_s_tok = h_s.reshape(B, H * W, C)          # (B, HW, C) for attention

    # ---- Global → spatial mixing (replaces bidirectional cross-attention) ----
    # h_v broadcasts a summary to all spatial locations so the spatial field
    # knows the global state before running FNO dynamics.
    # Spatial → global direction is handled by the GRU pool at the end of _core.
    h_v_proj = self.sub('mix_v2s', nn.Linear, C, **self.kw)(h_v)  # (B, C)
    h_s_tok = h_s_tok + h_v_proj[:, None, :]                       # (B, HW, C)
    h_s_tok = self.sub('mix_norm_s', nn.Norm, self.norm)(h_s_tok)

    h_s = h_s_tok.reshape(B, H, W, C)

    # ---- FNO-2D: PDE dynamics on h_spatial ----
    # rfft2 requires float32; cast up, compute, cast back
    m = min(self.fno_modes, H // 2 + 1, W // 2 + 1)
    h_f = h_s.astype(jnp.float32)
    h_fft = jnp.fft.rfft2(h_f, axes=(1, 2))                        # (B,H,W//2+1,C)
    h_fft_m = h_fft[:, :m, :m, :]                                   # (B,m,m,C)
    h_ri = jnp.concatenate(
        [h_fft_m.real, h_fft_m.imag], axis=-1)                      # (B,m,m,2C)
    h_ri = h_ri.reshape(B, m * m * 2 * C).astype(compute_dtype)
    h_ri = self.sub('fno_spec', nn.Linear, m * m * 2 * C, **self.kw)(h_ri)
    h_ri = h_ri.reshape(B, m, m, 2 * C).astype(jnp.float32)
    h_fft_new = jnp.zeros_like(h_fft).at[:, :m, :m, :].set(
        h_ri[:, :, :, :C] + 1j * h_ri[:, :, :, C:])
    h_spec = jnp.fft.irfft2(
        h_fft_new, s=(H, W), axes=(1, 2)).astype(compute_dtype)     # (B,H,W,C)

    # Local bypass: per-location channel mixing (W path in FNO)
    h_loc = self.sub('fno_loc', nn.Linear, C, **self.kw)(
        h_s_tok).reshape(B, H, W, C)

    # Forcing term: action broadcast to all spatial locations
    act_f = self.sub('fno_act', nn.Linear, C, **self.kw)(action)
    act_f = nn.act(self.act)(self.sub('fno_actnorm', nn.Norm, self.norm)(act_f))

    # Stochastic term: z broadcast to all spatial locations
    sto_f = self.sub('fno_sto', nn.Linear, C, **self.kw)(stoch_flat)
    sto_f = nn.act(self.act)(self.sub('fno_stonorm', nn.Norm, self.norm)(sto_f))

    # Combine: L[u] + W[u] + f(a) + H(z)
    cand_s_tok = (
        h_spec + h_loc +
        act_f[:, None, None, :] + sto_f[:, None, None, :]
    ).reshape(B, H * W, C)
    cand_s_tok = nn.act(self.act)(
        self.sub('fno_candnorm', nn.Norm, self.norm)(cand_s_tok))

    # Gated update for h_spatial (GRU-style stability)
    gate_s = jax.nn.sigmoid(
        self.sub('fno_gate', nn.Linear, C, **self.kw)(
            jnp.concatenate([h_s_tok, cand_s_tok], -1)) - 1)
    h_s_new = (
        gate_s * jnp.tanh(cand_s_tok) + (1 - gate_s) * h_s_tok
    ).reshape(B, H, W, C)

    # ---- GRU update for h_vec ----
    pool = h_s_new.mean(axis=(1, 2))    # (B, C) global average pool
    gru_in = jnp.concatenate([h_v, pool, action, stoch_flat], -1)
    gru_out = self.sub('vec_gru', nn.Linear, 3 * D, **self.kw)(gru_in)
    r, cand_v, upd = jnp.split(gru_out, 3, -1)
    r = jax.nn.sigmoid(r)
    cand_v = jnp.tanh(r * cand_v)
    upd = jax.nn.sigmoid(upd - 1)
    h_v_new = upd * cand_v + (1 - upd) * h_v

    return jnp.concatenate([h_s_new.reshape(B, sp), h_v_new], axis=-1)

  def _prior(self, feat):
    x = feat
    for i in range(self.imglayers):
      x = self.sub(f'prior{i}', nn.Linear, self.hidden, **self.kw)(x)
      x = nn.act(self.act)(self.sub(f'prior{i}norm', nn.Norm, self.norm)(x))
    return self._logit('priorlogit', x)

  def _logit(self, name, x):
    kw = dict(**self.kw, outscale=self.outscale)
    x = self.sub(name, nn.Linear, self.stoch * self.classes, **kw)(x)
    return x.reshape(x.shape[:-1] + (self.stoch, self.classes))

  def _dist(self, logits):
    out = embodied.jax.outs.OneHot(logits, self.unimix)
    out = embodied.jax.outs.Agg(out, 1, jnp.sum)
    return out
