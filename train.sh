python train.py \
        --cuda \
        -d coco \
        -root /mnt/share/ssd2/dataset \
        --batch_size 16 \
        --img_size 640 \
        --max_epoch 150 \
        --lr_drop 100 \
        --aux_loss \
        --use_nms \
        # --batch_first \
        --no_warmup
