# LLaVA-Med-3D: Applying LLaVa-Med for 3D Medical Visual Question Answering

### Firstly, install all necessary packages as in LLaVA-Med: https://github.com/microsoft/LLaVA-Med

### To run the model, put the folder vbm_images into LLaVA-Med-3D/dataset_3D first. Then:

- If choosing to load weight llava-med 1.5 and train stage 1 and stage 2, run script finetune_3D.sh
- If choosing to load weight llava-med 1.5 and directly train on instruction-tuning dataset (stage 2), run script finetune_3D_only_stage2.sh

### To inference the model, firstly you format the question.json as .json file you use to train the model and change paths (data, weight) in inference.sh file. Then, run:
```Shell
bash inference_3D.sh
```

### Code Details:

- To process a list of images (slices) for each question-answer, I load, preprocess, and store a tensor including all preprocessed slices for each QA in file llava/train/train.py, from line 720-777 for training,
and in llava/eval/model_vqa.py, from line 65-75 for inference. The class DataCollatorForSupervisedDataset in llava/train/train.py also help load data as batch.
- To combine list of preprocessed slices into language (QA), looking at file llava/model/llava_arch.py, from line 198-386 (function named prepare_inputs_labels_for_multimodal).
