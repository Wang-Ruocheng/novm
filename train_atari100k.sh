#!/bin/bash
# Atari 100k benchmark — NOWM
# 用法:
#   bash train_atari100k.sh                    # 跑全部 26 个游戏，顺序执行
#   bash train_atari100k.sh pong               # 只跑一个游戏
#   GAMES="pong breakout" bash train_atari100k.sh   # 跑指定游戏
#   GPU=1 bash train_atari100k.sh pong         # 指定 GPU 编号

set -e

LOGROOT="${LOGROOT:-logs/pong_nowm_v21}"
GPU="${GPU:-0}"
SEED="${SEED:-0}"
STEPS="${STEPS:-1.1e5}"

ALL_GAMES=(
  alien amidar assault asterix bank_heist battle_zone boxing breakout
  chopper_command crazy_climber demon_attack freeway frostbite gopher
  hero jamesbond kangaroo krull kung_fu_master ms_pacman pong
  private_eye qbert road_runner seaquest up_n_down
)

if [ -n "$1" ]; then
  GAMES=("$1")
elif [ -n "$GAMES" ]; then
  read -ra GAMES <<< "$GAMES"
else
  GAMES=("${ALL_GAMES[@]}")
fi

echo "Games: ${GAMES[*]}"
echo "Logroot: $LOGROOT"
echo "GPU: $GPU  Seed: $SEED"

for game in "${GAMES[@]}"; do
  echo ""
  echo "=== Starting: $game ==="
  CUDA_VISIBLE_DEVICES=$GPU python -m dreamerv3.main \
    --configs atari100k nowm \
    --task "atari100k_${game}" \
    --seed "$SEED" \
    --logdir "$LOGROOT/${game}_s${SEED}" \
    --run.steps "$STEPS" \
    --jax.platform cuda \
    "${@:2}"
  echo "=== Done: $game ==="
done

echo ""
echo "All games finished. Logs: $LOGROOT"
