#!/bin/bash

################################################################################
# PLASTIC: Enhancing Plasticity for First Session Adaptation Continual Learning
# Example launch commands for different datasets and methods
# 
# Usage: 
#   1. Uncomment the experiment you want to run
#   2. Adjust hyperparameters as needed
#   3. Run: bash launch.sh
#
# Note: Make sure to configure WandB (see README.md)
################################################################################

################################################################################
# CIFAR-100 Experiments (224x224 resolution for ViT)
################################################################################

## PLASTIC (Our Method) - Test-Time Adaptation with LayerNorm tuning
## 10 tasks, 10 classes per task, starting from 10 initial classes
python main.py --config ./exps/adapt_tta.json \
    --dataset cifar224 \
    --experiment_name PLASTIC_CIFAR100_10Tasks_SEED0 \
    --seed 0 \
    --convnet_type pretrained_vit_b16_224_in21k_adapter \
    --layers norm \
    --exp_grp PLASTIC_CIFAR100 \
    --increment 10 \
    --init_cls 10 \
    --selection_p 0.5 \
    --niter 1 \
    --N_augment 8 \
    --batch_tta \
    --random_aug \
    --start_TTA 0

## SimpleCIL Baseline (without Test-Time Adaptation)
## Same task configuration as PLASTIC for fair comparison
# python main.py --config ./exps/simplecil.json \
#     --dataset cifar224 \
#     --experiment_name SimpleCIL_CIFAR100_10Tasks_SEED0 \
#     --seed 0 \
#     --convnet_type pretrained_vit_b16_224_in21k \
#     --exp_grp SimpleCIL_CIFAR100 \
#     --increment 10 \
#     --init_cls 10

## FECAM Baseline (Forward Compatible Mean)
# python main.py --config ./exps/fecam.json \
#     --dataset cifar224 \
#     --experiment_name FECAM_CIFAR100_10Tasks_SEED0 \
#     --seed 0 \
#     --convnet_type pretrained_vit_b16_224_adapter \
#     --exp_grp FECAM_CIFAR100 \
#     --increment 10 \
#     --init_cls 10

## ADAM Baseline with Adapters
# python main.py --config ./exps/adam_adapter.json \
#     --dataset cifar224 \
#     --experiment_name ADAM_CIFAR100_10Tasks_SEED0 \
#     --seed 0 \
#     --convnet_type pretrained_vit_b16_224_in21k_adapter \
#     --exp_grp ADAM_CIFAR100 \
#     --increment 10 \
#     --init_cls 10

################################################################################
# CIFAR-100 Standard Resolution (32x32)
################################################################################

## PLASTIC on CIFAR-100 (32x32)
# python main.py --config ./exps/adapt_tta.json \
#     --dataset cifar100 \
#     --experiment_name PLASTIC_CIFAR100_32x32_SEED0 \
#     --seed 0 \
#     --convnet_type pretrained_vit_b16_224_in21k_adapter \
#     --layers norm \
#     --exp_grp PLASTIC_CIFAR100_32 \
#     --increment 10 \
#     --init_cls 10 \
#     --selection_p 0.5 \
#     --niter 1 \
#     --N_augment 8 \
#     --batch_tta \
#     --random_aug \
#     --start_TTA 0

################################################################################
# ImageNet-R Experiments (Paper's main benchmark)
################################################################################

## PLASTIC on ImageNet-R - 20 tasks, 10 classes per task
## This reproduces the ImageNet-R results from the paper
# python main.py --config ./exps/adapt_tta.json \
#     --dataset imagenetr \
#     --experiment_name PLASTIC_20Tasks_ImgnetR_SEED0 \
#     --seed 0 \
#     --convnet_type pretrained_vit_b16_224_in21k_adapter \
#     --layers norm \
#     --exp_grp PLASTIC_LongTasks \
#     --increment 10 \
#     --init_cls 10 \
#     --selection_p 0.5 \
#     --niter 1 \
#     --N_augment 8 \
#     --batch_tta \
#     --random_aug \
#     --start_TTA 0 \
#     --print_forget

## SimpleCIL Baseline on ImageNet-R
# python main.py --config ./exps/simplecil.json \
#     --dataset imagenetr \
#     --experiment_name SimpleCIL_20Tasks_ImgnetR_SEED0 \
#     --seed 0 \
#     --convnet_type pretrained_vit_b16_224_in21k \
#     --exp_grp SimpleCIL_ImgnetR \
#     --increment 10 \
#     --init_cls 10 \
#     --print_forget

################################################################################
# ImageNet-A Experiments
################################################################################

## PLASTIC on ImageNet-A
# python main.py --config ./exps/adapt_tta.json \
#     --dataset imageneta \
#     --experiment_name PLASTIC_ImgnetA_SEED0 \
#     --seed 0 \
#     --convnet_type pretrained_vit_b16_224_in21k_adapter \
#     --layers norm \
#     --exp_grp PLASTIC_ImgnetA \
#     --increment 10 \
#     --init_cls 10 \
#     --selection_p 0.5 \
#     --niter 1 \
#     --N_augment 8 \
#     --batch_tta \
#     --random_aug \
#     --start_TTA 0

################################################################################
# Robustness Experiments (Corrupted Datasets)
################################################################################

## PLASTIC on CIFAR-100-C (All corruptions, level 5)
# python main.py --config ./exps/adapt_tta.json \
#     --dataset cifar100-c \
#     --experiment_name PLASTIC_CIFAR100C_Robustness_SEED0 \
#     --seed 0 \
#     --convnet_type pretrained_vit_b16_224_in21k_adapter \
#     --layers norm \
#     --exp_grp PLASTIC_Robustness \
#     --increment 10 \
#     --init_cls 10 \
#     --selection_p 0.5 \
#     --niter 1 \
#     --N_augment 8 \
#     --batch_tta \
#     --random_aug \
#     --start_TTA 0

################################################################################
# Ablation Studies
################################################################################

## Ablation: Different adaptation layers
## Try different --layers options: norm (best), adapter, head, all

## LayerNorm only (recommended)
# python main.py --config ./exps/adapt_tta.json \
#     --dataset cifar224 \
#     --experiment_name PLASTIC_Ablation_LayerNorm \
#     --layers norm \
#     --seed 0 --increment 10 --init_cls 10

## Adapter modules only
# python main.py --config ./exps/adapt_tta.json \
#     --dataset cifar224 \
#     --experiment_name PLASTIC_Ablation_Adapter \
#     --layers adapter \
#     --seed 0 --increment 10 --init_cls 10

## All parameters
# python main.py --config ./exps/adapt_tta.json \
#     --dataset cifar224 \
#     --experiment_name PLASTIC_Ablation_All \
#     --layers all \
#     --seed 0 --increment 10 --init_cls 10

## Ablation: Different TTA start points
## --start_TTA 0 (from first task) vs --start_TTA 1 (from second task)

# python main.py --config ./exps/adapt_tta.json \
#     --dataset cifar224 \
#     --experiment_name PLASTIC_Ablation_TTA_Start1 \
#     --start_TTA 1 \
#     --seed 0 --layers norm --increment 10 --init_cls 10

################################################################################
# Multi-Seed Experiments (for statistical significance)
################################################################################

## Run with different seeds for robust evaluation
# for SEED in 0 1 2 3 4; do
#     python main.py --config ./exps/adapt_tta.json \
#         --dataset cifar224 \
#         --experiment_name PLASTIC_CIFAR100_SEED${SEED} \
#         --seed ${SEED} \
#         --convnet_type pretrained_vit_b16_224_in21k_adapter \
#         --layers norm \
#         --increment 10 \
#         --init_cls 10 \
#         --selection_p 0.5 \
#         --niter 1 \
#         --N_augment 8 \
#         --batch_tta \
#         --random_aug \
#         --start_TTA 0
# done

################################################################################
# Notes:
# 1. Adjust --batch_size if you encounter CUDA OOM errors
# 2. Configure WandB entity in main.py line 123 or disable WandB
# 3. Datasets are auto-downloaded to ./data/ (except ImageNet variants)
# 4. Use --print_forget to see forgetting metrics
# 5. Results are logged to ./logs/ and WandB
################################################################################
