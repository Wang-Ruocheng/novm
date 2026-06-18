#!/usr/bin/env python3
"""
NOWM 梯度流诊断 v2 — 包含 rec_loss。

完整训练循环：rec_loss（强迫 stoch 编码当前观测）+ rep/dyn_loss（KL 正则）。
核心逻辑：carry 用 prior_deter，decoder 用 concat(prior_deter, stoch)，
   -> prior_deter 没有当前观测，decoder 必须依赖 stoch
   -> rec_loss 提供了阻止 KL 塌缩的恢复力
"""
import os, sys
sys.path.insert(0, '/Users/ruochengwang/NOWM')
os.environ['JAX_PLATFORMS'] = 'cpu'

import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import elements
import optax
from dreamerv3 import wm as rssm
import embodied.jax.nets as nn

B, T = 4, 16
lat_size, lat_chan, deter_d = 4, 4, 8
C_enc = 16
tokens_dim = lat_size ** 2 * C_enc
act_space = {'action': elements.Space(np.int32, (), 0, 4)}

rng = jax.random.PRNGKey(0)
tokens  = jax.random.normal(rng, (B, T, tokens_dim))
action  = {'action': jnp.zeros((B, T), dtype=jnp.int32)}
reset   = jnp.zeros((B, T), dtype=bool)

nowm = rssm.NOWM(
    act_space,
    lat_size=lat_size, lat_chan=lat_chan, deter=deter_d,
    hidden=32, stoch=4, classes=8, fno_modes=2,
    obslayers=1, imglayers=1, act='silu', norm='rms',
    outscale=1.0, winit='trunc_normal_in',
    unimix=0.01, free_nats=1.0, absolute=False,
    name='nowm',
)

stoch_dim = nowm.stoch * nowm.classes  # 4 * 8 = 32
total_deter = nowm._total()            # lat_size^2 * lat_chan + deter

# ── 简单线性 decoder：concat(prior_deter, stoch_flat) → reconstruct tokens ──
decoder = nn.Linear(tokens_dim, name='dec')

# ── 初始化参数 ──────────────────────────────────────────────────────────────
def init_fn(tokens):
    carry = nowm.initial(B)
    _, _, losses, feat, _ = nowm.loss(carry, tokens, action, reset, training=False)
    # 初始化 decoder：用 feat 输出
    deter_flat = feat['deter'].reshape(B * T, -1)
    stoch_flat = feat['stoch'].reshape(B * T, -1)
    dec_in = jnp.concatenate([deter_flat, stoch_flat], axis=-1)
    _ = decoder(dec_in)
    return losses, feat

pure_init = nj.pure(init_fn)
state = {}
state, (losses0, feat0) = pure_init(state, tokens, seed=0, create=True)

print("=" * 60)
print("NOWM 梯度流诊断 v2（含 rec_loss）")
print("=" * 60)

# ── 1. 初始 KL ────────────────────────────────────────────────────────────
dyn_raw = float(losses0['dyn'].mean())
rep_raw = float(losses0['rep'].mean())
print(f"\n[1] 初始化后 KL: dyn={dyn_raw:.4f}, rep={rep_raw:.4f} nats")
print(f"    free_nats=1.0 → {'梯度可流动 ✓' if rep_raw > 1.0 else '梯度被截断 ✗'}")

# ── 2. 梯度流检查（仅 rep+dyn，基线）────────────────────────────────────
def loss_fn_baseline(state_, tokens_):
    carry = nowm.initial(B)

    def inner():
        _, _, losses, _, _ = nowm.loss(carry, tokens_, action, reset, training=False)
        return losses['rep'].mean() + losses['dyn'].mean()

    pure_inner = nj.pure(inner)
    state_out, total = pure_inner(state_, seed=0)
    return total, state_out

grads = jax.grad(lambda s: loss_fn_baseline(s, tokens)[0])(state)

obs_norms, prior_norms, other_norms = [], [], []
for k, g in grads.items():
    gnorm = float(jnp.linalg.norm(g.ravel()))
    if 'obslogit' in k or '/obs' in k:
        obs_norms.append(gnorm)
    elif 'prior' in k:
        prior_norms.append(gnorm)
    else:
        other_norms.append(gnorm)

def fmt(label, norms):
    if not norms:
        print(f"  {label}: (无参数)")
        return
    zeros = sum(1 for n in norms if n < 1e-10)
    print(f"  {label}: {len(norms)} params, "
          f"mean={np.mean(norms):.2e}, max={np.max(norms):.2e}, "
          f"零梯度={zeros}/{len(norms)}")

print(f"\n[2] rep+dyn loss 梯度（基线）:")
fmt("obslogit", obs_norms)
fmt("prior   ", prior_norms)
fmt("其他    ", other_norms)

# ── 3. 完整 loss（rec + rep + dyn）训练模拟 ───────────────────────────
print(f"\n[3] 模拟 20 步完整训练（rec + rep + dyn）:")

optimizer = optax.adam(3e-4)
opt_state = optimizer.init(state)

def full_loss_fn(state_, tokens_):
    """完整 loss：rec_loss 强迫 stoch 携带当前观测信息。"""
    carry = nowm.initial(B)

    def inner():
        _, _, losses, feat, _ = nowm.loss(carry, tokens_, action, reset, training=True)

        # rec_loss 代理：decoder 用 (prior_deter, stoch) 重建 tokens
        # prior_deter 没有当前观测 → decoder 必须依赖 stoch
        deter_flat = feat['deter'].reshape(B * T, -1)          # prior_deter
        stoch_flat = feat['stoch'].reshape(B * T, -1)
        dec_in = jnp.concatenate([deter_flat, stoch_flat], axis=-1)
        rec_out = decoder(dec_in)
        target = tokens_.reshape(B * T, tokens_dim).astype(rec_out.dtype)
        rec_loss = jnp.mean((rec_out - target) ** 2)

        total = rec_loss + losses['rep'].mean() + losses['dyn'].mean()
        kl = losses['rep'].mean()
        return total, kl

    pure_inner = nj.pure(inner)
    state_out, (total, kl) = pure_inner(state_, seed=0)
    return total, kl, state_out

kl_history = [rep_raw]
cur_state  = state
cur_opt    = opt_state

for step in range(20):
    t_ = tokens + jax.random.normal(
        jax.random.PRNGKey(step + 100), tokens.shape) * 0.1

    def step_loss(s):
        total, kl, s_out = full_loss_fn(s, t_)
        return total, (kl, s_out)

    (total, (kl_val, new_state)), grads_step = jax.value_and_grad(
        step_loss, has_aux=True)(cur_state)
    updates, cur_opt = optimizer.update(grads_step, cur_opt)
    cur_state = optax.apply_updates(new_state, updates)
    kl_history.append(float(kl_val))

print(f"  step  0: KL = {kl_history[0]:.4f}")
for i in [4, 9, 14, 19]:
    arrow = "↑" if kl_history[i+1] > kl_history[i] else "↓"
    print(f"  step {i+1:2d}: KL = {kl_history[i+1]:.4f}  {arrow}")

final_kl   = kl_history[-1]
initial_kl = kl_history[0]
trend = final_kl - initial_kl
print(f"\n  KL 变化: {initial_kl:.4f} → {final_kl:.4f}  (Δ={trend:+.4f})")

print()
if trend > 0.05:
    print(">>> ✓ KL 上升：rec_loss 提供了恢复力，stoch 学会携带观测信息。")
elif final_kl >= 1.0 and abs(trend) < 0.05:
    print(">>> ~ KL 稳定在 free_nats 附近：未明显塌缩，可观察更长时间。")
else:
    print(">>> ✗ KL 仍在塌缩。需进一步检查 rec_loss 权重或 outscale。")
