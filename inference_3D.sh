
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

python3 llava/eval/model_vqa.py \
	--model-path weights/llava_$DATASET-$EPOCH-epo$NOTE$NOTE_OUTPUT \
	--question-file $IMAGE_FOLDER/$DATASET/test_fix_brief.json  \
	--image-folder $IMAGE_FOLDER/$DATASET/${DATASET}_images \
	--answers-file results/captioning/${DATASET}$NOTE.jsonl \
	--conv-mode llava_llama_2 \
	--temperature 0.1 