#!/bin/bash
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=120:0:0    
#SBATCH --mail-user=rathodchaitanya11@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:a100:4

cd $sr_project
module purge
module load python/3.9.21 scipy-stack
module load opencv
source ~/projects/def-mhassanz/cjrathod/sr_env/bin/activate
# Set environment variables for DDP rendezvous
export TORCH_NCCL_ASYNC_HANDLING=1
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export NCCL_DEBUG=INFO
export MASTER_PORT=3456
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export WANDB_API_KEY= ""

echo "r$SLURM_NODEID master: $MASTER_ADDR"
echo "r$SLURM_NODEID Launching python script"
echo "Training Starting"
srun python main.py --init_method tcp://$MASTER_ADDR:3456 --world_size $((SLURM_NTASKS_PER_NODE * SLURM_JOB_NUM_NODES))
echo "Training Ended" 
