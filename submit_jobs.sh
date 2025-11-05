#!/bin/bash
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err
#SBATCH --partition=dgxh
#SBATCH --nodes=1
##SBATCH --nodelist=dgxh-1,dgxh-2,dgxh-3,dgxh-4
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --signal=B:TERM@60
#SBATCH --job-name=blackout-sudoku
#SBATCH --requeue

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

VERSION=1

SCHEDULE="blackout"
LOSS="blackout"
BATCH_SIZE=16
T_END=15
T=1000
LR=3e-4
GAMMA=0.75
MIN_LR=3e-4
NUM_EPOCHS=1000
NUM_ITERS=500000
SAVE_EVERY=10000
DISABLE_W=0
TIME_DIST="uniform"
LOAD_FROM="null" # load latest, for requeueing
NORMALIZE=0
WARMUP=1
WANDB=0

/nfs/hpc/share/vuonga2/conda-env/diff/bin/python bdsd_train.py \
  --load_from $LOAD_FROM \
  --version $VERSION \
  --normalize $NORMALIZE \
  --warmup $WARMUP \
  --wandb $WANDB \
  --schedule $SCHEDULE \
  --loss $LOSS \
  --batch_size $BATCH_SIZE \
  --tEnd $T_END \
  --T $T \
  --lr $LR \
  --gamma $GAMMA \
  --min_lr $MIN_LR \
  --num_epochs $NUM_EPOCHS \
  --num_iters $NUM_ITERS \
  --save_every $SAVE_EVERY \
  --disable_w $DISABLE_W \
  --time_dist $TIME_DIST