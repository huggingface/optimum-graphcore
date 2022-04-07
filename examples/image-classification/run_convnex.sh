python run_image_classification_on_local_data.py \
--train_dir /localdata/datasets/imagenet-raw-data/train  \
--train_val_split 0.1  --output_dir ./outputs/ --do_train  --do_eval --num_train_epochs 3 \
--learning_rate 3e-4     --per_device_train_batch_size 2  --per_device_eval_batch_size 1 \
--pod_type pod16     --dataloader_num_workers 8     --dataloader_drop_last  \
--seed 1337 --model_name_or_path facebook/convnext-tiny-224 --fp32 --disable_feature_extractor --disable_mixup