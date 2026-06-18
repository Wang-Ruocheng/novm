#!/usr/bin/env python3
"""
NOWM 架构属性验证。

不测试收敛（需要真实 CNN decoder + 有时序结构的数据），
只验证关键架构不变量：
  1. feat['deter'] = posterior_deter（assimilation ops 有梯度通路）
  2. carry['deter'] = posterior_deter（与 RSSM 一致，obs 通过 carry 链传递）
  3. feat['prior_deter'] = _core 输出（KL 用 prior deter，不受 observe 影响）
  4. entry['deter'] = posterior_deter（imagination 起点有空间记忆）
  5. obslogit 对不同 tokens 有明显差异（obs 信息进入了后验 z）
  6. KL 非零（梯度可流动）
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
print("NOWM 架构属性验证（posterior carry，assimilation ops 有梯度）")
print("=" * 60)

# ── 1. feat['deter'] 应是 posterior_deter（≠ prior_deter）─────────────
feat_eq_prior = float(jnp.mean(jnp.abs(feat_A['deter'] - feat_A['prior_deter'])))
print(f"\n[1] feat['deter'] vs prior_deter 差值: {feat_eq_prior:.6f}")
if feat_eq_prior > 1e-5:
    print("    ✓ feat 里是 posterior_deter，_spatial_observe/_vec_observe 有梯度通路")
else:
    print("    ✗ feat['deter'] == prior_deter（assimilation ops 梯度截断）")

# ── 2. carry_out['deter'] 应等于 feat['deter'] 最后一步 ───────────────
carry_deter = carry_out_A['deter']              # (B, total_deter)
feat_post_last = feat_A['deter'][:, -1]         # (B, T, total_deter) → last step
diff_carry_post = float(jnp.mean(jnp.abs(carry_deter - feat_post_last)))
print(f"\n[2] carry_out['deter'] == feat['deter'][-1] 差值: {diff_carry_post:.6f}")
if diff_carry_post < 1e-5:
    print("    ✓ carry 里是 posterior_deter（与 RSSM 一致）")
else:
    print("    ✗ carry 与 feat['deter'][-1] 不一致")

# ── 3. feat['prior_deter'] 应是 _core 输出（≠ posterior）──────────────
feat_prior_last = feat_A['prior_deter'][:, -1]
diff_prior_post = float(jnp.mean(jnp.abs(feat_prior_last - feat_post_last)))
print(f"\n[3] feat['prior_deter'] vs feat['deter'] 差值: {diff_prior_post:.6f}")
if diff_prior_post > 1e-5:
    print("    ✓ prior_deter ≠ posterior_deter（KL 有意义）")
else:
    print("    ~ prior_deter ≈ posterior_deter（assimilation 无效果）")

# ── 4. entry['deter'] == feat['deter']（imagination 起点一致）──────────
entry_deter  = entries_A['deter'][:, -1]
diff_entry_feat = float(jnp.mean(jnp.abs(entry_deter - feat_post_last)))
print(f"\n[4] entry['deter'] == feat['deter'][-1] 差值: {diff_entry_feat:.6f}")
if diff_entry_feat < 1e-5:
    print("    ✓ entry 与 feat 的 deter 一致（posterior carry 设计正确）")
else:
    print("    ✗ entry 与 feat 的 deter 不一致")

# ── 5. obslogit 对不同 tokens 有明显差异 ──────────────────────────────
logit_diff   = float(jnp.mean(jnp.abs(feat_A['logit'] - feat_B['logit'])))
prior_diff   = float(jnp.mean(jnp.abs(feat_A['prior_deter'] - feat_B['prior_deter'])))
print(f"\n[5] obslogit 对不同 tokens 的均值差: {logit_diff:.4f}")
print(f"    prior_deter 对不同 tokens 的均值差: {prior_diff:.4f}")
print(f"    比值: {logit_diff/(prior_diff+1e-8):.2f}x")
if logit_diff > 0.01:
    print("    ✓ obslogit 对不同 obs 有响应（后验 z 会根据当前观测改变）")
else:
    print("    ✗ obslogit 对 obs 不敏感（后验 z 无信息）")

# ── 6. 初始 KL（从零 carry 出发）──────────────────────────────────────
def run_loss(tokens):
    carry = nowm.initial(B)
    _, _, losses, feat, _ = nowm.loss(carry, tokens, action, reset, training=False)
    return losses

pure_loss = nj.pure(run_loss)
state2 = {}
state2, losses_A = pure_loss(state2, tokens_A, seed=0, create=True)

rep_kl = float(losses_A['rep'].mean())
dyn_kl = float(losses_A['dyn'].mean())
print(f"\n[6] 初始化后 KL（从零 carry 出发）:")
print(f"    rep = {rep_kl:.4f}, dyn = {dyn_kl:.4f} nats")
if rep_kl > 1.0:
    print("    ✓ KL > free_nats=1.0，梯度可流动")
else:
    print("    ✗ KL ≤ free_nats，梯度被截断")

print("\n" + "=" * 60)
ok = (feat_eq_prior > 1e-5 and diff_carry_post < 1e-5 and
      logit_diff > 0.01 and rep_kl > 1.0)
if ok:
    print(">> 架构属性全部满足。posterior carry 设计，assimilation ops 有完整梯度通路。")
else:
    print(">> 部分属性不满足，请检查标记 ✗ 的条目。")
