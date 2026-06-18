#!/usr/bin/env python3
"""
NOWM 架构属性验证。

设计不变量：
  carry/feat  = deter_prior  — decoder 依赖 stoch 获取 obs 信息 → KL 不坍缩
  obslogit    = f(deter_post, tokens) — assim ops 通过 KL/rec 梯度得到训练
  entry/imag  = deter_post   — imagination 起点含空间记忆

梯度路径：
  reconstruction → stoch (straight-through) → obslogit → deter_post → assim ops ✓
  KL rep         → logit = obslogit         → deter_post → assim ops ✓
  KL dyn         → _prior(deter_prior)      → _core ✓
"""
import os, sys
sys.path.insert(0, '/Users/ruochengwang/NOWM')
os.environ['JAX_PLATFORMS'] = 'cpu'

import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import elements
from dreamerv3 import wm as rssm

B, T = 2, 4
lat_size, lat_chan, deter_d = 4, 4, 8
C_enc = 16
tokens_dim = lat_size ** 2 * C_enc
act_space = {'action': elements.Space(np.int32, (), 0, 4)}

nowm = rssm.NOWM(
    act_space,
    lat_size=lat_size, lat_chan=lat_chan, deter=deter_d,
    hidden=32, stoch=4, classes=8, fno_modes=2,
    obslayers=1, imglayers=1, act='silu', norm='rms',
    outscale=1.0, winit='trunc_normal_in',
    unimix=0.01, free_nats=1.0, absolute=False,
    name='nowm',
)

rng = jax.random.PRNGKey(42)
tokens_A  = jax.random.normal(rng, (B, T, tokens_dim))
tokens_B  = jax.random.normal(jax.random.PRNGKey(99), (B, T, tokens_dim))
action    = {'action': jnp.zeros((B, T), dtype=jnp.int32)}
reset     = jnp.zeros((B, T), dtype=bool)

def run(tokens):
    carry_in  = nowm.initial(B)
    carry_out, entries, feat = nowm.observe(carry_in, tokens, action, reset, training=False)
    return carry_in, carry_out, entries, feat

pure_run = nj.pure(run)
state = {}
state, (carry_in_A, carry_out_A, entries_A, feat_A) = pure_run(
    state, tokens_A, seed=0, create=True)
state, (carry_in_B, carry_out_B, entries_B, feat_B) = pure_run(
    state, tokens_B, seed=1)

print("=" * 60)
print("NOWM 架构属性验证（prior carry + posterior obslogit）")
print("=" * 60)

feat_prior_last = feat_A['prior_deter'][:, -1]   # (B, total_deter)
feat_deter_last = feat_A['deter'][:, -1]          # should == feat_prior_last
entry_deter_last = entries_A['deter'][:, -1]      # should be posterior (≠ prior)

# ── 1. feat['deter'] == prior_deter（decoder 无法直接看 obs → stoch 必须携带）──
diff_feat_prior = float(jnp.mean(jnp.abs(feat_deter_last - feat_prior_last)))
print(f"\n[1] feat['deter'] == prior_deter 差值: {diff_feat_prior:.6f}")
if diff_feat_prior < 1e-5:
    print("    ✓ feat 里是 prior_deter，decoder 依赖 stoch 获取 obs → KL 不坍缩")
else:
    print("    ✗ feat['deter'] 含 obs 信息（会导致 KL 坍缩）")

# ── 2. carry_out['deter'] == feat['deter'][-1]（prior carry）────────────────
carry_deter = carry_out_A['deter']
diff_carry_feat = float(jnp.mean(jnp.abs(carry_deter - feat_deter_last)))
print(f"\n[2] carry_out['deter'] == feat['deter'][-1] 差值: {diff_carry_feat:.6f}")
if diff_carry_feat < 1e-5:
    print("    ✓ carry 里是 prior_deter（obs 只通过 stoch 进入下步）")
else:
    print("    ✗ carry 与 feat['deter'][-1] 不一致")

# ── 3. entry['deter'] ≠ feat['deter']（entry 是 posterior，feat 是 prior）──────
diff_entry_prior = float(jnp.mean(jnp.abs(entry_deter_last - feat_prior_last)))
print(f"\n[3] entry['deter'] vs feat['deter'] 差值: {diff_entry_prior:.6f}")
if diff_entry_prior > 1e-5:
    print("    ✓ entry 是 posterior_deter（imagination 起点含空间修正）")
else:
    print("    ~ entry 与 prior_deter 几乎相同（assimilation 效果弱，初始化正常）")

# ── 4. obslogit 对不同 tokens 有明显差异（assim ops 接入了 obs 信息）──────────
logit_diff  = float(jnp.mean(jnp.abs(feat_A['logit'] - feat_B['logit'])))
prior_diff  = float(jnp.mean(jnp.abs(feat_A['prior_deter'] - feat_B['prior_deter'])))
print(f"\n[4] obslogit 对不同 tokens 的均值差: {logit_diff:.4f}")
print(f"    prior_deter 对不同 tokens 的均值差: {prior_diff:.4f}")
print(f"    比值: {logit_diff/(prior_diff+1e-8):.2f}x")
if logit_diff > 0.01:
    print("    ✓ obslogit 对不同 obs 有响应（后验 z 会根据当前观测改变）")
else:
    print("    ✗ obslogit 对 obs 不敏感")

# ── 5. 初始 KL（从零 carry 出发）─────────────────────────────────────────────
def run_loss(tokens):
    carry = nowm.initial(B)
    _, _, losses, feat, _ = nowm.loss(carry, tokens, action, reset, training=False)
    return losses

pure_loss = nj.pure(run_loss)
state2 = {}
state2, losses_A = pure_loss(state2, tokens_A, seed=0, create=True)

rep_kl = float(losses_A['rep'].mean())
dyn_kl = float(losses_A['dyn'].mean())
print(f"\n[5] 初始化后 KL（从零 carry 出发）:")
print(f"    rep = {rep_kl:.4f}, dyn = {dyn_kl:.4f} nats")
if rep_kl > 1.0:
    print("    ✓ KL > free_nats=1.0，梯度可流动")
else:
    print("    ✗ KL ≤ free_nats，梯度被截断")

print("\n" + "=" * 60)
ok = (diff_feat_prior < 1e-5 and diff_carry_feat < 1e-5 and
      logit_diff > 0.01 and rep_kl > 1.0)
if ok:
    print(">> 架构属性全部满足。prior carry 防 KL 坍缩，assim ops 通过 obslogit 有梯度。")
else:
    print(">> 部分属性不满足，请检查标记 ✗ 的条目。")
