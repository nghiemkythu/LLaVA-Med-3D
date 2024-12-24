import os
import re
import sys
import pandas as pd, numpy as np
import json
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
import nltk.translate.meteor_score as meteor
# from score_eval import Meteor
# from CXRMetric.run_eval import cacl_metric 
# import pymeteor.pymeteor as pymeteor
import nltk
nltk.download('wordnet')

from tabulate import tabulate
import argparse
import logging
import warnings
# from pandas.core.common import SettingWithCopyWarning
# warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
# print(Meteor.get_score([["an apple is on tree"]], "there is an apple on tree"))
# (["this is a test"], ["this is a test"])

def _bioclean(caption):
    return re.sub('[.,?;*!%^&_+():-\[\]{}]',
                  '',
                  caption.replace('"', '')
                  .replace('/', '')
                  .replace('\\', '')
                  .replace("'", '')
                  .strip()
                  .lower()) 

def load_jsonl(path):
    data=[]
    with open(path, 'r', encoding='utf-8') as reader:
        for line in reader:
            data.append(json.loads(line))
    return data 

def sum_of_list(list_data):
    total = list()
    for i in list_data:
        total += i
    return total

class CaptionsEvaluation:

    def __init__(self, gold_dir, results_dir):
        self.gold_dir = gold_dir
        self.results_dir = results_dir
        self.gold_data = {}
        self.results_data = {}
    
    def load_data(self):
        # # gold_csv = pd.read_csv(self.gold_dir, 
        # #                        sep="\t", 
        # #                        header=None, 
        # #                        names=["image_ids", "captions"], 
        # #                        dtype=object)
        # self.gold_data = dict(zip(gold_csv["image_ids"], gold_csv["captions"]))
        with open(self.gold_dir) as f:
            self.gold_data = json.load(f)
        for item in self.gold_data:
            self.gold_data[item['id']] = item['conversations'][1]['value']
        result_data_list = load_jsonl(self.results_dir)
        for item in result_data_list:
            self.results_data[item['question_id']] = item['text']
        
    def preprocess_captions(self, images_caption):
        processed_caption = {}
        for i in range(len(images_caption)):
            processed_caption[i] = [_bioclean(images_caption[i])]
        return processed_caption

    def evaluate(self):
        self.load_data()
        gold_data = self.preprocess_captions(self.gold_data)
        print(gold_data[0])
        results_data = self.preprocess_captions(self.results_data)

        print(results_data[0])
        # Set up scorers
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            # (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr")
        ]
        # Compute score for each metric
        for scorer, method in scorers:
            # print("Computing", method, "...")
            # if method == "METEOR":

            #     # print(sum_of_list(gold_data.values())[:3])
            #     # meteor_score_list = list()
            #     # gold_data_value = list(gold_data.values())
            #     # results_data_value = list(results_data.values())
            #     # for i in range(len(gold_data)):
            #     #     meteor_score_list.append(Meteor.get_score(gold_data_value[i], results_data_value[i][0]))
            #     # score = float(np.mean(meteor_score_list))
            #     score = meteor.meteor_score(list(gold_data.values()),
            #         sum_of_list(results_data.values()))
            # else:
            score, scores = scorer.compute_score(gold_data, results_data)
            if type(method) == list:
                for sc, m in zip(score, method):
                    print("%s : %0.3f" % (m, sc))
            else:
                print("%s : %0.3f" % (method, score))

        # Compute CXR score
        # save_file = "result.csv"
        # cxr_score = cacl_metric(gold_data, results_data, save_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold-dir", type=str, 
                        default="/netscratch/duynguyen/Research/Nghiem_LLaVA-Med/Data_Report_Gen/iu_xray/test.json")
    parser.add_argument("--results-dir", type=str, 
                        default="/netscratch/duynguyen/Research/trung_LLaVA-Med/LVLM-Med/results/100_1e0_multi_graph_100_scale_test_bugfix_2e-5_gpt.jsonl")
                        # default="/netscratch/duynguyen/Research/LLaVA-Med/results_finetuned/iu_xray_captioning/ans-opt-small-100_1e0_multi_graph_100_scale_dci_test_bugfix_2e-5_brief.jsonl")
                        # default="results/captioning/iu_xray_stage2_3-epo_stage1_1-epo_og.jsonl")
    args = parser.parse_args()
    evaluator = CaptionsEvaluation(args.gold_dir, args.results_dir)
    evaluator.evaluate()
