#!/bin/bash
set -e

GPU=0

for DEPTH in 20 56 110
do
    for PERCENT in 0.2 0.4 0.6 0.8
    do
        CUDA_VISIBLE_DEVICES=$GPU python preact_resnet_cifar.py --dataset "svhn"  --tmp "tmp/SVHN" \
                             --epochs 40 --milestones "20,30" --retrain-epochs 40 --percent $PERCENT --depth $DEPTH
        
        CUDA_VISIBLE_DEVICES=$GPU python preact_resnet_cifar.py --dataset "svhn"  --tmp "tmp/SVHN" \
                             --epochs 40 --milestones "20,30" --retrain-epochs 40 --percent $PERCENT --depth $DEPTH
        
        CUDA_VISIBLE_DEVICES=$GPU python preact_resnet_cifar.py --dataset "svhn"  --tmp "tmp/SVHN" \
                             --epochs 40 --milestones "20,30" --retrain-epochs 40 --percent $PERCENT --depth $DEPTH
        
        CUDA_VISIBLE_DEVICES=$GPU python preact_resnet_cifar.py --dataset "svhn"  --tmp "tmp/SVHN" \
                             --epochs 40 --milestones "20,30" --retrain-epochs 40 --percent $PERCENT --depth $DEPTH

        CUDA_VISIBLE_DEVICES=$GPU python preact_resnet_cifar.py --dataset "svhn"  --tmp "tmp/SVHN" \
                             --epochs 40 --milestones "20,30" --retrain-epochs 40 --percent $PERCENT --depth $DEPTH
        
        CUDA_VISIBLE_DEVICES=$GPU python preact_resnet_cifar.py --dataset "svhn"  --tmp "tmp/SVHN" \
                             --epochs 40 --milestones "20,30" --retrain-epochs 40 --percent $PERCENT --depth $DEPTH
        
        CUDA_VISIBLE_DEVICES=$GPU python preact_resnet_cifar.py --dataset "svhn"  --tmp "tmp/SVHN" \
                             --epochs 40 --milestones "20,30" --retrain-epochs 40 --percent $PERCENT --depth $DEPTH
        
        CUDA_VISIBLE_DEVICES=$GPU python preact_resnet_cifar.py --dataset "svhn"  --tmp "tmp/SVHN" \
                             --epochs 40 --milestones "20,30" --retrain-epochs 40 --percent $PERCENT --depth $DEPTH

        CUDA_VISIBLE_DEVICES=$GPU python preact_resnet_cifar.py --dataset "svhn"  --tmp "tmp/SVHN" \
                             --epochs 40 --milestones "20,30" --retrain-epochs 40 --percent $PERCENT --depth $DEPTH
        
        CUDA_VISIBLE_DEVICES=$GPU python preact_resnet_cifar.py --dataset "svhn"  --tmp "tmp/SVHN" \
                             --epochs 40 --milestones "20,30" --retrain-epochs 40 --percent $PERCENT --depth $DEPTH
        
    done
done
