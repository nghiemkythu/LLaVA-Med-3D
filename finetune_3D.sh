
DATASET="VLM_ADNI_DATA"
NOTE=original_LLaVA-Med_new
NOTE_OUTPUT="_pre-train_stage_1_3D_mlp" # lora: with adapter again, non: no adapter.
DATASET_LINK="/netscratch/duynguyen/Research/bao_llava_med/Dense/dataset_3D/$DATASET"

MODEL_NAME="/netscratch/duynguyen/Research/LLaVA-Med/weights_full/checkpoint_llava_med_instruct_60k_inline_mention_version_1-5"
#VISION_TOWER=goog§le/siglip-so400m-patch14-384
VISION_TOWER=openai/clip-vit-large-patch14

EPOCH=2
TRAIN_BATCH_SIZE=4
EVAL_BATCH_SIZE=4
STEP=4

# torchrun --nnodes=1 --nproc_per_node=4 --master_port=25100 llava/train/train_mem.py \
# 	--model_name_or_path $MODEL_NAME \
# 	--deepspeed scripts/zero2.json \
# 	--version llava_llama_2 \
# 	--data_path $DATASET_LINK/AD_caption-llava_3D_version.json \
# 	--image_folder $DATASET_LINK/vbm_images \
#    	--vision_tower $VISION_TOWER \
# 	--tune_mm_mlp_adapter True \
# 	--mm_dense_connector_type None \
# 	--mm_projector_type mlp2x_gelu \
# 	--mm_vision_select_layer -2 \
# 	--mm_use_im_start_end False \
# 	--mm_use_im_patch_token False \
# 	--bf16 True \
# 	--output_dir weights/llava_$DATASET-$EPOCH-epo$NOTE$NOTE_OUTPUT \
# 	--num_train_epochs $EPOCH \
# 	--per_device_train_batch_size $TRAIN_BATCH_SIZE \
# 	--per_device_eval_batch_size $EVAL_BATCH_SIZE \
# 	--gradient_accumulation_steps $STEP \
# 	--evaluation_strategy "no" \
# 	--save_strategy "steps" \
# 	--save_steps 100 \
# 	--save_total_limit 1 \
# 	--learning_rate 2e-3 \
# 	--weight_decay 0. \
# 	--warmup_ratio 0.03 \
# 	--lr_scheduler_type "cosine" \
# 	--logging_steps 1 \
# 	--tf32 True \
# 	--model_max_length 8192 \
# 	--gradient_checkpointing True \
# 	--dataloader_num_workers 4 \
# 	--lazy_preprocess True \
# 	--report_to wandb \
# 	--run_name $DATASET-$EPOCH-epo$NOTE$NOTE_OUTPUT

# sleep 5s

###################### STAGE 2 ############################

NOTE_OUTPUT_2="_pre-train_stage_2_3D_mlp_origin_LLaVA" 
TRAIN_BATCH_SIZE=4
EVAL_BATCH_SIZE=4
STEP=8

torchrun --nnodes=1 --nproc_per_node=4 --master_port=25100 llava/train/train_mem.py \
	--model_name_or_path /netscratch/duynguyen/Research/LLaVA-Med/weights/LLaVA-7b-v0 \
	--deepspeed scripts/zero3.json \
	--version llava_llama_2 \
	--data_path $DATASET_LINK/AD_instruct_post_3D_version.json \
	--image_folder $DATASET_LINK/vbm_images \
   	--vision_tower $VISION_TOWER \
	--mm_dense_connector_type None \
	--mm_projector_type mlp2x_gelu \
	--tune_mm_mlp_adapter False \
	--group_by_modality_length True \
	--mm_vision_select_layer -2 \
	--mm_use_im_start_end False \
	--mm_use_im_patch_token False \
	--bf16 True \
	--output_dir weights/llava_$DATASET-3-epo$NOTE$NOTE_OUTPUT_2 \
	--num_train_epochs 3 \
	--per_device_train_batch_size $TRAIN_BATCH_SIZE \
	--per_device_eval_batch_size $EVAL_BATCH_SIZE \
	--gradient_accumulation_steps $STEP \
	--evaluation_strategy "no" \
	--save_strategy "steps" \
	--save_steps 50000 \
	--save_total_limit 1 \
	--learning_rate 2e-5 \
	--weight_decay 0. \
	--warmup_ratio 0.03 \
	--lr_scheduler_type "cosine" \
	--logging_steps 1 \
	--tf32 True \
	--model_max_length 8192 \
	--gradient_checkpointing True \
	--dataloader_num_workers 4 \
	--lazy_preprocess True \
	--report_to wandb \
	--run_name $DATASET-3-epo$NOTE$NOTE_OUTPUT_2

