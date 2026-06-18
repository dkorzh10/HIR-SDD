#!/bin/bash
# Run a job locally
# Usage: ./manage/run_locally.sh <config_path> [gpu_ids]
# Example: ./manage/run_locally.sh config.yaml
# Example: ./manage/run_locally.sh config.yaml 0,1  # Override auto-detection

ARG1=$1
MANUAL_GPU_IDS=$2

if [ -z "$ARG1" ]; then
    echo "Usage: $0 <config_path> [gpu_ids]"
    echo "  config_path: Path to config file OR experiment name (e.g. qwen_audio_sft)"
    echo "  gpu_ids: (Optional) Comma-separated GPU IDs to override auto-detection"
    echo ""
    echo "GPUs are automatically selected based on config's Runner.num_gpus setting"
    exit 1
fi

# Resolve config path (same logic as run_mls_job.sh)
if [ -f "$ARG1" ]; then
    CONFIG_FILE="$ARG1"
    EXPERIMENT_NAME=$(basename $(dirname "$CONFIG_FILE"))
elif [ -d "experiment_configs/$ARG1" ]; then
    EXPERIMENT_NAME="$ARG1"
    CONFIG_FILE="experiment_configs/$EXPERIMENT_NAME/config.yaml"
else
    echo "Error: Config file or experiment directory '$ARG1' not found"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file '$CONFIG_FILE' not found"
    exit 1
fi

# Get number of GPUs from config
NUM_GPUS_CONFIG=$(awk '/Runner:/ {flag=1; next} /^[A-Z]/ {flag=0} flag && /num_gpus:/ {print $2; exit}' "$CONFIG_FILE")
if [ -z "$NUM_GPUS_CONFIG" ]; then
    NUM_GPUS_CONFIG=1
fi

# Check total available GPUs
TOTAL_AVAILABLE=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$TOTAL_AVAILABLE" -eq 0 ]; then
    echo "Error: No GPUs found (nvidia-smi failed)"
    exit 1
fi

# Function to find available GPUs
find_available_gpus() {
    local needed=$1
    
    # Get GPU utilization using nvidia-smi
    # Format: GPU_ID UTILIZATION% MEMORY_USED MEMORY_TOTAL
    local gpu_stats=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits)
    
    local available_gpus=()
    local count=0
    
    while IFS=, read -r gpu_id util mem_used mem_total; do
        # Trim whitespace
        gpu_id=$(echo "$gpu_id" | xargs)
        util=$(echo "$util" | xargs)
        mem_used=$(echo "$mem_used" | xargs)
        mem_total=$(echo "$mem_total" | xargs)
        
        # Consider GPU available if utilization < 10% AND memory usage < 10%
        if [ "$util" -lt 10 ] && [ "$mem_used" -lt $((mem_total / 10)) ]; then
            available_gpus+=($gpu_id)
            count=$((count + 1))
            if [ $count -eq $needed ]; then
                break
            fi
        fi
    done <<< "$gpu_stats"
    
    if [ $count -lt $needed ]; then
        return 1
    fi
    
    # Return comma-separated list
    IFS=,
    echo "${available_gpus[*]}"
}

# Determine which GPUs to use
if [ -n "$MANUAL_GPU_IDS" ]; then
    # Manual override provided
    GPU_IDS="$MANUAL_GPU_IDS"
    NUM_GPUS=$(echo $GPU_IDS | tr ',' '\n' | wc -l)
    
    # Validate manually specified GPUs exist
    for GPU in $(echo $GPU_IDS | tr ',' ' '); do
        if [ $GPU -ge $TOTAL_AVAILABLE ]; then
            echo "Error: GPU $GPU not available (only $TOTAL_AVAILABLE GPUs: 0-$((TOTAL_AVAILABLE-1)))"
            exit 1
        fi
    done
    
    echo "Using manually specified GPUs: $GPU_IDS"
else
    # Auto-detect available GPUs
    NUM_GPUS=$NUM_GPUS_CONFIG
    
    if [ $NUM_GPUS -gt $TOTAL_AVAILABLE ]; then
        echo "Error: Config requests $NUM_GPUS GPUs but only $TOTAL_AVAILABLE available"
        exit 1
    fi
    
    echo "Auto-detecting $NUM_GPUS available GPU(s)..."
    GPU_IDS=$(find_available_gpus $NUM_GPUS)
    
    if [ -z "$GPU_IDS" ]; then
        echo "Error: Could not find $NUM_GPUS free GPU(s)"
        echo ""
        echo "Current GPU status:"
        nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,name --format=table
        echo ""
        echo "You can manually specify GPUs: $0 $ARG1 <gpu_ids>"
        exit 1
    fi
    
    echo "Selected GPUs: $GPU_IDS (based on lowest utilization)"
fi

export PYTHONPATH=$PYTHONPATH:$(pwd)
export CUDA_VISIBLE_DEVICES=$GPU_IDS
# Reduce CUDA allocator fragmentation (see https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Setup logging
LOG_DIR="logs/local_runs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/${EXPERIMENT_NAME}_${TIMESTAMP}.log"

echo "Running training with config: $CONFIG_FILE"
echo "Using GPUs: $GPU_IDS ($NUM_GPUS total)"
echo "Logging to: $LOG_FILE"
echo "To stop: ./manage/stop_locally.sh $EXPERIMENT_NAME"

# Activate virtual environment if it exists (preferred)
if [ -f ".venv/bin/python3" ]; then
    echo "Using virtual environment at .venv"
    PYTHON_CMD=".venv/bin/python3"
    TORCHRUN_CMD=".venv/bin/torchrun"
    # Ensure bin is in PATH for subprocesses
    export PATH=".venv/bin:$PATH"
else
    echo "Warning: .venv not found. Using system python."
    PYTHON_CMD="python3"
    TORCHRUN_CMD="torchrun"
fi

# Function to find an available port
find_available_port() {
    local start_port=${1:-29500}
    local max_tries=100
    
    for i in $(seq 0 $max_tries); do
        local port=$((start_port + i))
        # Check if port is in use
        if ! ss -tuln 2>/dev/null | grep -q ":$port " && ! netstat -tuln 2>/dev/null | grep -q ":$port "; then
            echo $port
            return 0
        fi
    done
    
    # Fallback: use random port in high range
    echo $((29500 + RANDOM % 1000))
}

# Run command and pipe output to both stdout and log file
if [ $NUM_GPUS -gt 1 ]; then
    # Find available port for distributed training
    MASTER_PORT=$(find_available_port 29500)
    echo "Using master port: $MASTER_PORT"
    
    $TORCHRUN_CMD --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT -m src.runner --config "$CONFIG_FILE" 2>&1 | tee "$LOG_FILE"
else
    $PYTHON_CMD -m src.runner --config "$CONFIG_FILE" 2>&1 | tee "$LOG_FILE"
fi