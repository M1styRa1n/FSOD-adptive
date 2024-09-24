#!/bin/bash
#SBATCH --job-name=fsod_coco
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=12G
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu  
#SBATCH --mail-user=hkong5@sheffield.ac.uk
#SBATCH --output=./Output/small_output_%j_%Y-%m-%d-%H-%M-%S.txt 
#SBATCH --mail-type=BEGIN,END,FAIL


# module spider cuda
module load Java/11.0.20
module load Anaconda3/2022.05
module load CUDA/11.1.1-GCC-10.2.0



source activate paper1
python -c "import torch; print(torch.cuda.is_available())"

export DETECTRON2_DATASETS="./dataset"
EXP_TAG=`[ -z "$1" ] && echo "$(date +%s)" || echo "$1"`
CKPT_DIR="checkpoints/coco/${EXP_TAG}"
NGPU=2

. ./hashmap.sh

MAX_ITER=$(create_hashmap)
STEPS=$(create_hashmap)

trap "exit" INT
trap "cleanup $MAX_ITER && cleanup $STEPS" exit

add_item "$MAX_ITER" 1 1600

add_item "$STEPS" 1 "(2560, )"


python3 main.py --num-gpus $NGPU --config-file configs/coco/base.yaml \
    MODEL.WEIGHTS "./pretrain/R-101.pkl" \
    OUTPUT_DIR "${CKPT_DIR}/base" \
    SEED 0
python3 -m tools.ckpt_surgery -d coco -m init -s 0 \
    "${CKPT_DIR}/base${split}/model_final.pth"
for seed in `seq 1 2`
do
    for shot in 1 
    do
        python3 main.py --num-gpus $NGPU --config-file configs/coco/fsod.yaml \
            DATASETS.TRAIN "('coco_2014_trainval_all_${shot}shot_seed${seed}', )" \
            MODEL.WEIGHTS "${CKPT_DIR}/base/model_final-fsod.pth" \
            OUTPUT_DIR "${CKPT_DIR}/fsod/${shot}shot/seed${seed}" \
            SOLVER.MAX_ITER "$(get_item ${MAX_ITER} ${shot})" \
            SOLVER.STEPS "$(get_item ${STEPS} ${shot})" \
            SEED $seed
    done
done

python -m tools.display_results "$CKPT_DIR"