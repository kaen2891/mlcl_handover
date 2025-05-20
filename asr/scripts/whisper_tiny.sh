
MODEL="whisper-tiny"
SEED="1"

for s in $SEED
do
    for m in $MODEL
    do
        TAG="bs8_lr5e-5_ep50_seed${s}"
        CUDA_VISIBLE_DEVICES=0 python main.py --tag $TAG \
                                        --dataset librispeech \
                                        --seed $s \
                                        --epochs 10 \
                                        --batch_size 8 \
                                        --train_annotation_file train-clean-100.csv \
                                        --dev_clean_annotation_file dev-other.csv \
                                        --dev_other_annotation_file dev-other.csv \
                                        --test_clean_annotation_file test-clean.csv \
                                        --test_other_annotation_file test-other.csv \
                                        --optimizer adam \
                                        --learning_rate 5e-5 \
                                        --weight_decay 1e-6 \
                                        --cosine \
                                        --model $m \
                                        --ma_update \
                                        --ma_beta 0.5 \
                                        --method ce \
                                        --print_freq 100

                                        # only for evaluation, add the following arguments
                                        # --eval \
                                        # --pretrained \
                                        # --pretrained_ckpt ./save/icbhi_ast_ce_bs8_lr5e-5_ep50_seed1/best.pth

    done
done
