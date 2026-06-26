#!/bin/bash
# Atari 100k benchmark — NOWM
# 用法:
#   bash train_atari100k.sh                    # 跑全部 26 个游戏，每游戏 3 seeds，顺序执行
#   bash train_atari100k.sh pong               # 只跑一个游戏，3 seeds
#   GAMES="pong breakout" bash train_atari100k.sh   # 跑指定游戏
#   GPU=1 bash train_atari100k.sh pong         # 指定 GPU 编号
#   SEEDS="0" bash train_atari100k.sh pong     # 只跑 seed 0

set -e

LOGROOT="${LOGROOT:-logs/pong_nowm_v23}"
GPU="${GPU:-0}"
SEEDS="${SEEDS:-0 1 2}"
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

read -ra SEED_LIST <<< "$SEEDS"

echo "Games: ${GAMES[*]}"
echo "Seeds: ${SEED_LIST[*]}"
echo "Logroot: $LOGROOT"
echo "GPU: $GPU"

for game in "${GAMES[@]}"; do
  for seed in "${SEED_LIST[@]}"; do
    echo ""
    echo "=== Starting: $game  seed=$seed ==="
    CUDA_VISIBLE_DEVICES=$GPU python -m dreamerv3.main \
      --configs atari100k nowm \
      --task "atari100k_${game}" \
      --seed "$seed" \
      --logdir "$LOGROOT/${game}_s${seed}" \
      --run.steps "$STEPS" \
      --jax.platform cuda \
      "${@:2}"
    echo "=== Done: $game  seed=$seed ==="
  done
done

echo ""
echo "All runs finished. Logs: $LOGROOT"
