export MASTER_ADDR="10.82.124.31"  # 还是节点0的IP
export MASTER_PORT="5678"
export NODE_COUNT="4"
export NODE_RANK="2"
export PROC_PER_NODE="8"
export RUN_NAME="memory_world_multinode"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${SCRIPT_DIR}/multi_nodes.sh"
