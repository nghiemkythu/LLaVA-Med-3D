import os
import argparse
import json
import pickle

def parse_option():
    parser = argparse.ArgumentParser('Processing Instruction Tuning for Stage 2', add_help=False)
    parser.add_argument('--path', type=str, default="/netscratch/duynguyen/Research/LLaVA-Med/data/instruct/", help='Path to raw data files', ) # /netscratch/duynguyen/Research/LLaVA-Med/data/instruct/llava_med_instruct_new_inline_mention.json
    parser.add_argument('--file', type=str, default="llava_med_instruct_new_inline_mention.json", help='llava_med_instruct_new_inline_mention.json', )
    args = parser.parse_args()
    return args
args = parse_option()

data_path = args.path + args.file 

instruction_prompt = """You are a helpful medical assistant. You are required to answer the question based on the medical image <image>.
Question: {question}."""

def instruction_tuning():
    print(f'Processing Stage 2 - Instruction tuning:')
    f = open(data_path)
    data = json.load(f)
    content = []
    for record in data:
        instune = {}
        instune['id'] = record['id']
        instune['image'] = record['image']
        instune['domain'] = record['domain']
        conversations = list()
        for conv in record['conversations']:
            if conv['from'] == 'human' and '<image>' in conv['value']:
                human_question = conv['value'].replace('\n','').replace('<image>','')
                human_question = instruction_prompt.format(question = human_question)
                conversations.append({'from': 'human', 'value': human_question})
            else:
                conversations.append(conv)
        instune['conversations'] = conversations

        content.append(instune)
    
    file_name = args.path + 'llava_med_instruction_tuning.json'
    with open(file_name, 'w') as f_saved:
        json.dump(content, f_saved)
    f.close()
    print('DONE!')

instruction_tuning()
   
