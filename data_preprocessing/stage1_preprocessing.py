import os
import argparse
import json
import pickle

def parse_option():
    parser = argparse.ArgumentParser('Processing Medical Alignment for Stage 1', add_help=False)
    parser.add_argument('--path', type=str, default="/netscratch/duynguyen/Research/LLaVA-Med/data/alignment/", help='Path to raw data files', ) # /netscratch/duynguyen/Research/LLaVA-Med/data/alignment/llava_med_alignment_new.json
    parser.add_argument('--file', type=str, default="llava_med_alignment_new.json", help='llava_med_alignment_new.json', )
    args = parser.parse_args()
    return args
args = parse_option()

data_path = args.path + args.file 

#instruction_prompt = """You are a helpful medical assistant. You are required to answer the question based on the medical image <image>.
#Question: {question}."""
instruction_prompt = """You are a helpful medical assistant.
You are given the medical image <image> and the following request.

### Request: {request}"""


def instruction_tuning():
    print(f'Processing Stage 1 - Medical Alignment:')
    f = open(data_path)
    data = json.load(f)
    content = []
    for record in data:
        instune = {}
        instune['id'] = record['id']
        instune['image'] = record['image']
       # instune['domain'] = record['domain']
        conversations = list()
        for conv in record['conversatons']:
            if conv['from'] == 'human' and '<image>' in conv['value']:
                human_question = conv['value'].replace('\n','').replace('<image>','')
                human_question = instruction_prompt.format(request = human_question)
                conversations.append({'from': 'human', 'value': human_question})
            else:
                conversations.append(conv)
        instune['conversatons'] = conversations

        content.append(instune)
    
    file_name = args.path + 'llava_med_alignment_prompt.json'
    with open(file_name, 'w') as f_saved:
        json.dump(content, f_saved)
    f.close()
    print('DONE!')

instruction_tuning()
   
