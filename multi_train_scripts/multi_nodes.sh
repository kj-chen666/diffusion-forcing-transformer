#!/bin/bash
set -euo pipefail

# === 基础环境设置 ===
# export PATH=/data/cuda/cuda-11.8/cuda/bin:$PATH
# export LD_LIBRARY_PATH=/data/cuda/cuda-11.8/cuda/lib64:$LD_LIBRARY_PATH
source /m2v_intern/chenkaijin/miniconda3/bin/activate
conda activate dfot
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"

# === 必需环境变量 ===
: "${PROC_PER_NODE:?PROC_PER_NODE is required}"
: "${MASTER_ADDR:?MASTER_ADDR is required}"
MASTER_PORT="${MASTER_PORT:-5678}"
: "${NODE_COUNT:?NODE_COUNT is required}"
: "${NODE_RANK:?NODE_RANK is required}"
RUN_NAME="${RUN_NAME:-memory_world_multinode}"

# === 关键修正：使用共享目录 ===
# 所有节点挂载的共享目录路径
# SHARED_DIR="/mnt/zhouxin-mnt/memory_world_model"  # 修改为您的实际共享路径
RENDEZVOUS_BASE="/m2v_intern/chenkaijin/multi_node_log"

# 创建唯一的任务ID（避免冲突）
JOB_KEY="job_${MASTER_ADDR}_${MASTER_PORT}_${NODE_COUNT}"
RV_DIR="${RENDEZVOUS_BASE}/${JOB_KEY}"
READY_FILE="${RV_DIR}/rank_${NODE_RANK}.ready"

# 创建目录
mkdir -p "$RV_DIR"

# 清理之前的就位文件（避免旧文件干扰）
# rm -f "$RV_DIR"/rank_*.ready

# 创建就位标志
touch "$READY_FILE"
echo "节点 $NODE_RANK 已就位：$READY_FILE"

# 等待所有节点就位
WAIT_POLL_SEC=5
MAX_WAIT_SEC=600  # 最多等待10分钟
start_time=$(date +%s)

echo "等待其他节点就位..."
while true; do
    ready_count=$(ls -1 "$RV_DIR"/rank_*.ready 2>/dev/null | wc -l | awk '{print $1}')
    current_time=$(date +%s)
    elapsed_time=$((current_time - start_time))
    
    echo "[$(date '+%H:%M:%S')] 已就位: ${ready_count}/${NODE_COUNT} 节点 (等待: ${elapsed_time}s)"
    
    if [[ "$ready_count" -eq "$NODE_COUNT" ]]; then
        echo "✅ 所有 $NODE_COUNT 个节点已就位！"
        break
    fi
    
    # 超时检查
    if [[ $elapsed_time -gt $MAX_WAIT_SEC ]]; then
        echo "❌ 等待超时（${MAX_WAIT_SEC}秒），已就位 ${ready_count}/${NODE_COUNT} 节点"
        echo "当前就位节点："
        ls -1 "$RV_DIR"/rank_*.ready 2>/dev/null || echo "无"
        exit 1
    fi
    
    sleep "$WAIT_POLL_SEC"
done

# === 启动训练 ===
echo "启动训练命令..."

# A800优化设置
# export NCCL_P2P_DISABLE=1
# export NCCL_SHM_DISABLE=1
# export NCCL_TIMEOUT=1800
# export NCCL_DEBUG=INFO

torchrun --nproc_per_node "$PROC_PER_NODE" \
  --master_addr "$MASTER_ADDR" \
  --master_port "$MASTER_PORT" \
  --nnodes "$NODE_COUNT" \
  --node_rank "$NODE_RANK" \
  -m main \
  +name="${RUN_NAME}" \
  dataset=memory_world \
  experiment=video_generation \
  algorithm=dfot_video_pose \
  load=null \
  wandb.mode=offline \
  experiment.num_nodes="${NODE_COUNT}" \
  experiment.training.checkpointing.every_n_train_steps=1000 \
  experiment.training.checkpointing.every_n_epochs=null

# 训练完成后清理
rm -f "$READY_FILE"
echo "节点 $NODE_RANK 训练完成"
