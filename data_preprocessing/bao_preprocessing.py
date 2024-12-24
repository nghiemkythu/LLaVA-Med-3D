import os
import argparse
import json
import pickle
def parse_option():
    parser = argparse.ArgumentParser('Processing VQA-RAD', add_help=False)
    parser.add_argument('--path', type=str, default="", help='Path to raw data files', )
    parser.add_argument('--dataset', choices=['vqa-rad', 'data_RAD', 'Slake1.0', 'pvqa'], default = 'test', help='Downstream data', )
    parser.add_argument('--original_file', default = '', help='Orignal data file', )
    parser.add_argument('--split', choices=['train', 'val', 'test'], default = 'test', help='Processing test or train dataset', )
    # parser.add_argument('--saved_name', help='prompt1, prompt2,...', )
    args = parser.parse_args()
    return args

args = parse_option()

path = args.path + args.dataset
# general_candidate = 'You are a helpful medical assistant. Your mission is to answer the following question based on the given image <image> and only return the answer as short as possible.'
#general_candidate = """### Instruction: You are a helpful medical assistant in answering the question based on the given medical image <image>. You ONLY return the best answer in a short way and give some approriate reasons for your answer. DO NOT return any irrelevant information."""
general_candidate = """### Instruction: You are a helpful medical assistant in answering the questions based on the given biomedical image <image>."""

end_candidate = """Here are some rules of the evaluation:
(1) You should prioritize evaluating whether the output honestly/precisely/closely executes the instruction.
(2) Outputs should NOT contain more/less than what the instruction asks for, as such outputs do NOT precisely execute the instruction."""

end_candidate1 =  """### Requirements for Your Output:
(1) The answer should *specifically* target the given instruction instead of some general standards, so the answers may revolve around key points of the instruction.
(2) You should directly give the answer with the rationale corresponding and PLEASE DONOT REPEAT THE QUESTION AGAIN. 
(3) Answers are presented without the wrong answers."""

note_closed_candidate = '(Answer "Yes" or "No" or the option mentioned in the question)'
note_open_candidate = '(Answer accurately the question)'

#closed_candidate = open_candidate = ''
#closed_candidate = open_candidate = '\nYour task is to answer correctly the following question with the medical knowledge and the given description.'
closed_candidate = open_candidate = ' Your task is to answer CORRECTLY the given questions with your medical knowledge.'
saved_name = 'p24'

question1 = "What is the image about?" 
question2 = "How do you explain your answer?" 
question3 = "Is there any unusual in the given image?"
question4 = "Another assistant whose answer is '{text}' for this question '{question}', what is your answer?"

def load_jsonl(path):
    data=[]
    with open(path, 'r', encoding='utf-8') as reader:
        for line in reader:
            data.append(json.loads(line))
    return data

def process_data_RAD():
  print(f'Processing data_RAD: {args.original_file}')
  description_f = load_jsonl('/netscratch/duynguyen/Research/bao_llava_med/Dense/results/inference_description_task2_data_RAD_stage2_3-epo_stage1_1-epo_dci64.jsonl')
  #print(description_f)
  #description = json.load(description_f)
  description = {int(d_i["question_id"]):d_i["text"].replace('\n', '') for d_i in description_f}
  #print(description)
  f = open(path + '/' + args.original_file)
  data = json.load(f)
  content = []
  for vqa in data:    
    #Create candidate answers for closed-set questions
    # Most(90%) of closed-set questions are [yes, no] type, but there are also many other type of closed-set (i.e [left, right], [up, down], etc.)
    # With non-yes-no type, we have to type the candidate answers directly from command line.  
    if vqa['answer_type'] == 'CLOSED': 
#        candidate_answers = general_candidate + closed_candidate + note_closed_candidate 
        candidate_answers = general_candidate + closed_candidate 
    else:
#        candidate_answers = general_candidate + open_candidate + note_open_candidate 
        candidate_answers = general_candidate + open_candidate 
    #End creating candidate answers
    vqa_instruct = {}
    vqa_instruct['id'] = int(vqa['qid'])
    vqa_instruct['image'] = vqa['image_name']
    vqa_instruct['answer_type'] = vqa['answer_type']
    note = note_closed_candidate if vqa['answer_type'] == 'CLOSED' else note_open_candidate 
    vqa_instruct['conversations'] = [{
      'from': 'human',
      'value': candidate_answers + f'\n{end_candidate1}' + '\n###Question 1: ' + question1 + '\n###Question 2: ' + vqa['question'] + '\n### Question 3: ' + question2 
#      'value': candidate_answers + f'\n###Question: ' + question4.format(text = description[int(vqa['qid'])], question = vqa['question']) 
#      'value': candidate_answers + f'\n###{end_candidate}' + '\n###Question: ' + vqa['question'] + ' ' + question2  
#    'value': candidate_answers + '\n###Description: ' + description[int(vqa['qid'])] + '\n###Question: ' + vqa['question']
#      'value': candidate_answers + '\n###Question: ' + question1 + ' ' +question3 
#      'value': vqa['question'] + candidate_answers + '\n<image>'
#     'value': candidate_answers + '\n\n' + end_candidate + '\n### Question: ' + vqa['question'] + '<image>'
#    'value': candidate_answers  + f'\n###Main Question: ' + vqa['question'] + '\n###Question 1: ' + question1 + '\n###Question 2: ' + question2
    },
      {
        'from': 'gpt',
        'value': str(vqa['answer'])
        }]
    content.append(vqa_instruct)
  file_name = f'test_w_options_{saved_name}.json' if 'test' in args.original_file else f'train_w_options_{saved_name}.json'
  with open(path + '/' + file_name, 'w') as f_saved: 
    json.dump(content, f_saved)
  f.close()

def process_vqa_rad(): 
  pass
def process_slake(): 
  print(f'Processing Slake: {args.original_file}')
  description_f = load_jsonl('/netscratch/duynguyen/Research/bao_llava_med/Dense/results/inference_p2_Slake1.0_new-15-epo_stage2_3-epo_stage1_1-epo_dci64.jsonl')
  #print(description_f)
  #description = json.load(description_f)
  description = {int(d_i["question_id"]):d_i["text"].replace('\n', '') for d_i in description_f}
  f = open(path + '/' + args.original_file)
  data = json.load(f)
  content = []
  for vqa in data:
    if vqa['q_lang'] != 'en': 
      continue 
    #Create candidate answers for closed-set questions
    # Most(90%) of closed-set questions are [yes, no] type, but there are also many other type of closed-set (i.e [left, right], [up, down], etc.)
    # With non-yes-no type, we have to type the candidate answers directly from command line.  
    if vqa['answer_type'] == 'CLOSED': 
        candidate_answers = general_candidate + closed_candidate + note_closed_candidate 
    else: 
        candidate_answers = general_candidate + open_candidate + note_open_candidate 
    #End creating candidate answers
    vqa_instruct = {}
    vqa_instruct['id'] = int(vqa['qid'])
    vqa_instruct['image'] = vqa['img_name']
    vqa_instruct['answer_type'] = vqa['answer_type']
    vqa_instruct['conversations'] = [{
      'from': 'human',
      'value': candidate_answers + f'\n{end_candidate1}' + '\n###Question 1: ' + question1 + '\n###Question 2: ' + vqa['question'] + '\n### Question 3: ' + question2 
      #'value': candidate_answers  + f'\n###Main Question: ' + vqa['question'] + '\n###Question 1: ' + question1 + '\n###Question 2: ' + question2
      #'value': candidate_answers + f'\n###Question: ' + question4.format(text = description[int(vqa['qid'])], question = vqa['question']) 
      #'value': candidate_answers + f'\n###{end_candidate}' + '\n###Question: ' + vqa['question'] + ' ' + question2
      #'value': candidate_answers + '\n### Question: ' + vqa['question']
      #'value': '<image>\n' + vqa['question'] 
      },
      {
        'from': 'gpt',
        'value': str(vqa['answer'])
        }]
    content.append(vqa_instruct)
  file_name = f'test_w_options_{saved_name}.json' if 'test' in args.original_file else f'train_w_options_{saved_name}.json'
  with open(path + '/' + file_name, 'w') as f_saved: 
    json.dump(content, f_saved)
  f.close()

def process_pvqa(): 
  print(f'Processing PathVQA: {args.original_file}')
  description_f = load_jsonl('/netscratch/duynguyen/Research/bao_llava_med/Dense/results/inference_p2_pvqa_new-15-epo_stage2_3-epo_stage1_1-epo_dci64.jsonl')
  #print(description_f)
  #description = json.load(description_f)
  description = {d_i["question_id"]:d_i["text"].replace('\n', '') for d_i in description_f}
  with open(path + '/' + args.original_file, 'rb') as f: # args.path: /netscratch/trnguyen/pvqa/qas/train_vqa.pkl
    data = pickle.load(f)
  content = []
  for vqa in data:    
    if vqa['answer_type'] == 'yes/no': 
      candidate_answers = general_candidate + closed_candidate + note_closed_candidate
    else:  
      candidate_answers = general_candidate + open_candidate + note_open_candidate 

    vqa_instruct = {}
    vqa_instruct['id'] = vqa['img_id']
    vqa_instruct['image'] = os.path.join(args.split, vqa['img_id'] + '.jpg')
    vqa_instruct['answer_type'] = 'CLOSED' if vqa['answer_type'] == 'yes/no' else 'OPEN'
    vqa_instruct['conversations'] = [{
      'from': 'human',
      'value': candidate_answers + f'\n{end_candidate1}' + '\n###Question 1: ' + question1 + '\n###Question 2: ' + vqa['sent'] + '\n### Question 3: ' + question2 
      #'value': candidate_answers  + f'\n###Main Question: ' + vqa['sent'] + '\n###Question 1: ' + question1 + '\n###Question 2: ' + question2
      #'value': candidate_answers + f'\n###Question: ' + question4.format(text = description[vqa['img_id']], question = vqa['sent']) 
      #'value': candidate_answers + f'\n###{end_candidate}' + '\n###Question: ' + vqa['sent'] + ' ' + question2
      #'value': candidate_answers + '\n### Question: ' +  vqa['sent'] 
      #'value': '<image>\n' + vqa['sent']
      },
      {
        'from': 'gpt',
        'value': str(next(iter(vqa['label'])))
        }]
    content.append(vqa_instruct)
  file_name = f'test_w_options_{saved_name}.json' if 'test' in args.original_file else f'train_w_options_{saved_name}.json'
  with open(path + '/' + file_name, 'w') as f_saved: 
    json.dump(content, f_saved)
  print('Done.')

if args.dataset == 'data_RAD': 
  process_data_RAD()
elif args.dataset == 'vqa-rad': 
  process_vqa_rad()
elif args.dataset == 'Slake1.0': 
  process_slake()
elif args.dataset == 'pvqa': 
  process_pvqa()
else: 
  raise ValueError("Unknown data, please choose from the following options: ['vqa-rad', 'data_RAD', 'Slake1.0', 'pathvqa']")


'''
Convert VQA-RAD to (one-round) conversation-style format. Dataset is saved as a list with following structure: 
[
  {
    "id": "17506892_F1",
    "image": "17506892_F1.jpg",
    "conversatons": [
      {
        "from": "human",
        "value": "<image>\nCan you describe the image for me?"
      },
      {
        "from": "gpt",
        "value": "The image consists of maps of significant voxels representing regions of hypoperfusion in FTLD patients according to their clinical diagnosis. These maps are superimposed onto a reference T1-weighted MRI image. There are five rows, each representing a different patient subgroup: bvFTD, SD, PNFA, PSP, and CBD. The neurological convention is followed, with the left side of the brain on the left side of the image."
      }
    ]
  }, 
  {
    "id":..., 
    "image": ..., 
    "conversation": ...
  
  }
'''


#Create candidate answer from all opened-set answers in training set 

#Candidate file has format: 
#{'0': [str, str, ..., str]} str: answer in training set, as string. 

'''
import json 

f = open(args.path)
data = json.load(f)
candidate_answers = {'0':[]}
for vqa in data: 
  if vqa['answer_type'] == 'OPEN' and vqa['phrase_type']  in ['freeform', 'para']: 
    answer = vqa['answer']
    if isinstance(answer, str):
      answer = answer.lower()
    if answer not in candidate_answers['0']:
      candidate_answers['0'].append(answer)
with open('candidate.json', 'w') as f_candidate: 
  json.dump(candidate_answers, f_candidate)
f.close()
'''


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#Single-round conversation. Each QA is an independent element in the list. 
# Use this (not the mult-round version) for converting format of VQA-RAD TEST DATA to conversation-style
'''
import json
from num2words import num2words
f = open(args.path)
data = json.load(f)
content = []
for vqa in data:
  if args.split == 'test':
     if vqa['phrase_type'] not in ['test_freeform', 'test_para']:
        continue
  else:
     if  vqa['phrase_type'] not in ['freeform', 'para']: 
        continue
    
  #Create candidate answers for closed-set questions
  # Most(90%) of closed-set questions are [yes, no] type, but there are also many other type of closed-set (i.e [left, right], [up, down], etc.)
  # With non-yes-no type, we have to type the candidate answers directly from command line.  
  #if vqa['answer_type'] == 'CLOSED': 
  #  if vqa['answer'].lower() in ['yes', 'no']: 
  #    candidate_answers = ' Please choose from the following two options: [yes,no]'
  #  else:
  #    print("'{}' is not a question with answers [yes,no] but {}".format(vqa['question'], vqa['answer']))
  #    print("Please type each candidate answers here. Enter new line after each candidate. Ctrl-D or Ctrl-Z ( windows ) to save it.")
  #    contents =[]
  #    while True:
  #      try:
  #        line = input()
  #      except EOFError:
  #        break
  #      line = line.strip()
  #      contents.append(line)
  #    candidate_answers = ' Please choose from the following {} options: [{}]'.format(num2words(len(contents)), ','.join(contents))
  #    print("Prompt: '{}'".format(candidate_answers))
  #    print()
  #else: 
  candidate_answers = ''
  #End creating candidate answers
  vqa_instruct = {}
  vqa_instruct['id'] = int(vqa['qid'])
  vqa_instruct['image'] = vqa['image_name']
  vqa_instruct['answer_type'] = vqa['answer_type']
  vqa_instruct['conversations'] = [{
    'from': 'human',
    'value': "<image>\n" + vqa['question'] + candidate_answers
    },
    {
      'from': 'gpt',
      'value': vqa['answer']
      }]
  content.append(vqa_instruct)
if args.split == 'train': 
  for vqa in data: 
     if vqa['question_frame']!= 'NULL': 
      vqa_instruct = {}
      vqa_instruct['id'] = int(vqa['qid'])
      vqa_instruct['image'] = vqa['image_name']
      vqa_instruct['answer_type'] = vqa['answer_type']
      vqa_instruct['conversations'] = [{
         'from': 'human',
         'value': "<image>\n" + vqa['question_frame']
         },
         {
            'from': 'gpt',
            'value': vqa['answer']
            }]
      content.append(vqa_instruct) 
file_name = 'test.json' if args.split == 'test' else 'train.json'
with open(file_name, 'w') as f_saved: 
   json.dump(content, f_saved)
 
f.close()
'''

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#Multi-round conversation. All QAs from the same image are grouped together
# Use this for converting format of VQA-RAD TRAIN DATA to conversation-style
'''
import json
from num2words import num2words
f = open(args.path)
data = json.load(f)

train = []
name_to_pos_train = {} # e.g {"synpic54610": 0, "synpic50962":1, "synpic34947":2, ... }
pos_train= 0

for vqa in data: 
  if vqa['phrase_type']  not in ['freeform', 'para']:
    continue 
  #Create candidate answers for closed-set questions
  # Most(90%) of closed-set questions are [yes, no] type, but there are also many other type of closed-set (i.e [left, right], [up, down], etc.)
  # With non-yes-no type, we have to type the candidate answers directly from command line.  
  if vqa['answer_type'] == 'CLOSED': 
    if vqa['answer'].lower() in ['yes', 'no']: 
      candidate_answers = ' Please choose from the following two options: [yes,no]'
    else:
      print("'{}' is not a question with answers [yes,no] but {}".format(vqa['question'], vqa['answer']))
      print("Please type each candidate answers here. Enter new line after each candidate. Ctrl-D or Ctrl-Z ( windows ) to save it.")
      contents =[]
      while True:
        try:
          line = input()
        except EOFError:
          break
        line = line.strip()
        contents.append(line)
      candidate_answers = ' Please choose from the following {} options: [{}]'.format(num2words(len(contents)), ','.join(contents))
      print("Prompt: '{}'".format(candidate_answers))
      print()
  else: 
    candidate_answers = ''
  #End creating candidate answers
  image_name = vqa['image_name'][:-4]
  if image_name in name_to_pos_train: 
    train[name_to_pos_train[image_name]]['conversations'].extend([{"from": "human", "value": vqa['question'] + candidate_answers}, {"from": "gpt", "value": vqa['answer']}])
  else: 
    vqa_instruct = {}
    vqa_instruct['id'] = int(vqa['qid'])
    vqa_instruct['image'] = vqa['image_name']
    vqa_instruct['conversations'] = [{
        "from": "human",
        "value": "<image>\n" + vqa['question'] + candidate_answers
      },
      {
        "from": "gpt",
        "value": vqa['answer']
      }]
    train.append(vqa_instruct)
    name_to_pos_train[image_name] = pos_train
    pos_train +=1
with open('train.json', 'w') as f_train:
    json.dump(train, f_train)
f.close()
'''


        
    
    
