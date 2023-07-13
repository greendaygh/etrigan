#!/bin/bash

python main.py --mode train --num_domains 2 --w_hpf 1 \
               --lambda_reg 1 --lambda_sty 1 --lambda_ds 1 --lambda_cyc 1 \
               --total_iters 10000 --batch_size 2 --eval_every 50000000 \
	       --train_img_dir data/celeba_hq/train \
	       --val_img_dir data/celeba_hq/val
