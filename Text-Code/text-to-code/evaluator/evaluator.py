# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.
import os
import logging
import argparse
from bleu import _bleu
import json
import sys
import pylcs
import evaluate
from rouge import Rouge

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def edit_distance(hyps,refs):
    overall_ED = 0
    num_elem = len(hyps)
    for elem in list(zip(hyps,refs)):
      tmp = pylcs.edit_distance(elem[0], elem[1])
      res_norm = 1-(tmp/max(len(elem[0]),len(elem[1])))
      overall_ED += res_norm
    return overall_ED/num_elem*100

def meteor(hyps, refs):
    meteor = evaluate.load('meteor')
    res_meteor = meteor.compute(predictions=hyps, references=refs)
    return res_meteor['meteor'] #* 100

def rouge(hyps, refs):
	metrics = ["rouge-1","rouge-2","rouge-3","rouge-4","rouge-l"]
	rouge = Rouge(metrics=metrics)
	scores = rouge.get_scores(hyps, refs, avg=True)
	f1_scores = (scores["rouge-1"]["f"], scores["rouge-2"]["f"], scores["rouge-3"]["f"], scores["rouge-4"]["f"], scores["rouge-l"]["f"])
	return f1_scores


def main():
    parser = argparse.ArgumentParser(description='Evaluate leaderboard predictions for code completion (line level).')
    parser.add_argument('--answers', '-a', required=True, help="filename of the labels, in json format.")
    parser.add_argument('--predictions', '-p', required=True, help="filename of the leaderboard predictions, in txt format.")
    args = parser.parse_args()

    preds = [elem.strip() for elem in open(args.predictions, "r").readlines()]
    gts = [elem.strip() for elem in open(args.answers, "r").readlines()]

    assert len(preds) == len(gts), f"Samples of predictions and answers are not equal, {len(preds)}: {len(gts)}"

    total = len(gts)
    EM = 0.0
    for pred, gt in zip(preds, gts):
        if pred.split() == gt.split():
            EM += 1

    for i in range(1,5):
        bleu_score = round(_bleu(args.answers, args.predictions, max_order = i), 2)
        logger.info(f"BLEU-{i}: {bleu_score}")
        
    logger.info(f"EM: {round(EM/total*100, 2)}")

    list_gold = gts
    list_answer = preds

    overall_ED = edit_distance(list_gold, list_answer)
    overall_Meteor = meteor(list_gold, list_answer)
    overall_Rouge = rouge(list_gold, list_answer)
    num_elem = len(list_answer)

    print(f"ED: {overall_ED}")
    print('METEOR:{0:.2f}\n'.format(overall_Meteor * 100))
    metrics = ["ROUGE-1","ROUGE-2","ROUGE-3","ROUGE-4","ROUGE-L"]
    for i, metric in enumerate(metrics):
        print(metric+':{0:.2f}\n'.format(overall_Rouge[i] * 100))
    

if __name__ == "__main__":
    main()
