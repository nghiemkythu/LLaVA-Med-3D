# Adapting multi-modal LLMs to 3D structure data

## 1. LLaVA-Med-3D: Applying LLaVa-Med for 3D Medical Visual Question Answering
### Firstly, install all necessary packages as in LLaVA-Med: https://github.com/microsoft/LLaVA-Med

### To run the model, put the folder vbm_images into LLaVA-Med-3D/dataset_3D first. Then:

- If choosing to load weight llava-med 1.5 and train stage 1 and stage 2, run script finetune_3D.sh
- If choosing to load weight llava-med 1.5 and directly train on instruction-tuning dataset (stage 2), run script finetune_3D_only_stage2.sh

## 2. MedFlamingo-3D: We modify MedFlamingo to make it work for 3D volume input.
