#!/usr/bin/env python3
"""
NOWM 架构属性验证。

不测试收敛（需要真实 CNN decoder + 有时序结构的数据），
只验证关键架构不变量：
  1. feat['deter'] = prior_deter（carry fix 后 feat 里无当前观测）
  2. carry['deter'] = prior_deter（obs 只能通过 stoch 瓶颈进入下一步）
  3. obslogit 对不同 tokens 有明显差异（obs 信息进入了后验 z）
  4. entry['deter'] = posterior_deter（imagination 起点保留空间记忆）
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
print("NOWM 架构属性验证（carry = prior_deter fix）")
print("=" * 60)

# ── 1. feat['deter'] 应等于 prior_deter ───────────────────────────────
feat_eq_prior = float(jnp.mean(jnp.abs(feat_A['deter'] - feat_A['prior_deter'])))
print(f"\n[1] feat['deter'] == prior_deter 差值: {feat_eq_prior:.6f}")
if feat_eq_prior < 1e-5:
    print("    ✓ feat 里是 prior_deter，stoch 是解码器唯一的当前观测来源")
else:
    print("    ✗ feat['deter'] 含有当前观测（会导致 KL 塌缩）")

# ── 2. carry_out['deter'] 应等于 prior_deter ──────────────────────────
# carry_out['deter'] 是序列最后一步的 carry，即 prior_deter[-1]
# 验证方式：carry_out['deter'] 与 carry_in 经 _core 的输出一致
# 即 carry_out['deter'] ≈ feat['prior_deter'][:, -1]
carry_deter = carry_out_A['deter']              # (B, total_deter)
feat_prior_last = feat_A['prior_deter'][:, -1]  # (B, T, total_deter) → last step
diff_carry_prior = float(jnp.mean(jnp.abs(carry_deter - feat_prior_last)))
print(f"\n[2] carry_out['deter'] == feat['prior_deter'][-1] 差值: {diff_carry_prior:.6f}")
if diff_carry_prior < 1e-5:
    print("    ✓ carry 里是 prior_deter，不含当前观测（obs 只通过 stoch 进入下步）")
else:
    print("    ✗ carry 里含有当前观测信息")

# ── 3. entry['deter'] ≠ prior_deter（应是 posterior_deter）─────────────
entry_deter  = entries_A['deter'][:, -1]        # last step entry
diff_entry_prior = float(jnp.mean(jnp.abs(entry_deter - feat_prior_last)))
print(f"\n[3] entry['deter'] vs prior_deter 差值: {diff_entry_prior:.6f}")
if diff_entry_prior > 0.01:
    print("    ✓ entry 里是 posterior_deter（imagination 起点含空间记忆）")
else:
    print("    ~ entry 与 prior_deter 几乎相同（spatial observe 效果弱）")

# ── 4. obslogit 对不同 tokens 有明显差异 ──────────────────────────────
logit_diff   = float(jnp.mean(jnp.abs(feat_A['logit'] - feat_B['logit'])))
prior_diff   = float(jnp.mean(jnp.abs(feat_A['prior_deter'] - feat_B['prior_deter'])))
print(f"\n[4] obslogit 对不同 tokens 的均值差: {logit_diff:.4f}")
print(f"    prior_deter 对不同 tokens 的均值差: {prior_diff:.4f}")
print(f"    比值: {logit_diff/(prior_diff+1e-8):.2f}x")
if logit_diff > 0.01:
    print("    ✓ obslogit 对不同 obs 有响应（后验 z 会根据当前观测改变）")
else:
    print("    ✗ obslogit 对 obs 不敏感（后验 z 无信息）")

# ── 5. 初始 KL（从零 carry 出发）──────────────────────────────────────
def run_loss(tokens):
    carry = nowm.initial(B)
    _, _, losses, feat, _ = nowm.loss(carry, tokens, action, reset, training=False)
    return losses

pure_loss = nj.pure(run_loss)
state2 = {}
state2, losses_A = pure_loss(state2, tokens_A, seed=0, create=True)
state2, losses_B = pure_loss(state2, tokens_B, seed=1)

rep_kl = float(losses_A['rep'].mean())
dyn_kl = float(losses_A['dyn'].mean())
print(f"\n[5] 初始化后 KL（从零 carry 出发）:")
print(f"    rep = {rep_kl:.4f}, dyn = {dyn_kl:.4f} nats")
if rep_kl > 1.0:
    print("    ✓ KL > free_nats=1.0，梯度可流动")
else:
    print("    ✗ KL ≤ free_nats，梯度被截断")

print("\n" + "=" * 60)
ok = (feat_eq_prior < 1e-5 and diff_carry_prior < 1e-5 and
      logit_diff > 0.01 and rep_kl > 1.0)
if ok:
    print(">> 架构属性全部满足。carry 修复有效，obs 只通过 stoch 进入动力学。")
    print(">> 实际训练中 CNN decoder 会强迫 stoch 携带观测信息，KL 应保持非零。")
else:
    print(">> 部分属性不满足，请检查标记 ✗ 的条目。")
