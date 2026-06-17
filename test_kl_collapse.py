#!/usr/bin/env python3
"""
NOWM KL 塌缩修复后验证。
修复：obslogit 和 feat['deter'] 改用 prior_deter（与 RSSM 一致）。
"""
import os, sys
sys.path.insert(0, '/Users/ruochengwang/NOWM')
os.environ['JAX_PLATFORMS'] = 'cpu'

import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import elements
from dreamerv3 import rssm
import embodied.jax.nets as nn

B, T = 4, 8
lat_size, lat_chan, deter_d = 4, 4, 8
C_enc = 16
tokens_dim = lat_size ** 2 * C_enc
act_space = {'action': elements.Space(np.int32, (), 0, 4)}

rng = jax.random.PRNGKey(42)
tokens_A = jax.random.normal(rng, (B, T, tokens_dim))
tokens_B = jax.random.normal(jax.random.PRNGKey(99), (B, T, tokens_dim))
action   = {'action': jnp.zeros((B, T), dtype=jnp.int32)}
reset    = jnp.zeros((B, T), dtype=bool)

nowm = rssm.NOWM(
    act_space,
    lat_size=lat_size, lat_chan=lat_chan, deter=deter_d,
    hidden=32, stoch=4, classes=8, fno_modes=2,
    obslayers=1, imglayers=1, act='silu', norm='rms',
    outscale=1.0, winit='trunc_normal_in',
    unimix=0.01, free_nats=0.0,   # free_nats=0 → 直接看原始 KL
    absolute=False,
    name='nowm',
)

def run_loss(tokens):
    carry = nowm.initial(B)
    _, _, losses, feat, _ = nowm.loss(carry, tokens, action, reset, training=False)
    return losses, feat

pure_loss = nj.pure(run_loss)
state = {}
state, (losses_A, feat_A) = pure_loss(state, tokens_A, seed=0, create=True)
state, (losses_B, feat_B) = pure_loss(state, tokens_B, seed=1)

print("=" * 60)
print("NOWM KL 塌缩修复后验证")
print("=" * 60)

diff_rel = float(jnp.mean(jnp.abs(feat_A['deter'] - feat_A['prior_deter'])) /
                 (jnp.mean(jnp.abs(feat_A['prior_deter'])) + 1e-8))
print(f"\n[1] feat['deter'] vs prior_deter 相对差: {diff_rel:.4f}")
print(f"    → 修复后应 ≈ 0（两者相同）")

dyn_kl = float(losses_A['dyn'].mean())
rep_kl = float(losses_A['rep'].mean())
print(f"\n[2] raw KL (free_nats=0): dyn={dyn_kl:.4f}, rep={rep_kl:.4f} nats")
print(f"    → 修复后应 > 1.0，与 free_nats=1.0 有余量，梯度可流动")

obs_sens = float(jnp.mean(jnp.abs(feat_A['logit'] - feat_B['logit'])))
prior_sens = float(jnp.mean(jnp.abs(feat_A['prior_deter'] - feat_B['prior_deter'])))
print(f"\n[3] obslogit 对 obs 变化的灵敏度: {obs_sens:.4f}")
print(f"    prior_deter 对 obs 变化的灵敏度: {prior_sens:.4f}")
print(f"    比值: {obs_sens/(prior_sens+1e-8):.2f}x")
print(f"    → 修复后比值应 >> 1（obslogit 主动使用 tokens）")

print()
if diff_rel < 0.01 and rep_kl > 1.1:
    print(">>> 修复有效：feat['deter']=prior_deter，KL 高于 free_nats 阈值。")
elif rep_kl > 1.0:
    print(">>> 修复基本有效：KL 在阈值以上。")
else:
    print(">>> 仍有问题，需进一步检查。")
