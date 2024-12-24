import pandas as pd 
import json
import argparse

# id, image, tag, conversation[{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]

instruction = """Describe briefly the X-ray image."""
# instruction2 = """Illustrate the image through a descriptive explanation <image>."""

def load_csv(path):
    data = pd.read_csv(path, 
                       sep="\t", 
                       header=None, 
                       names=["image_ids", "captions"], 
                       dtype=object)
    return data 

def preprocessing(path, file):
    file_path = f"{path}/{file}_images.tsv"
    data = load_csv(file_path)
    data_gen = json.load(open(f"{path}/iu_xray_captions.json"))
    print(len(data_gen))
    content = list()
    print(len(data))
    for i in range(len(data)):
        vqa = {}
        vqa['id'] = i 
        vqa['image'] = data['image_ids'][i]
    #print(data_gen[data['image_ids'][i]])
        vqa['conversations'] = [
            {"from": "human", "value": instruction}, 
            {"from": "gpt", "value": data_gen[data['image_ids'][i]]}
        ]
        content.append(vqa)
    save_path = f"{path}/{file}_prompt2.json"
    with open(save_path, 'w') as f:
       json.dump(content, f)
    print("DONE!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess data for image captioning evaluation")
    parser.add_argument("--path", type=str, help="Path to the gold data", 
                        default="/netscratch/duynguyen/Research/Nghiem_LLaVA-Med/Data_Report_Gen/iu_xray")
    parser.add_argument("--file", type=str, help="train/test", default="test")
    
    args = parser.parse_args()
    print("Preprocessing data...", args.file)
    preprocessing(args.path, args.file)
    