import argparse
import json
import collections
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from llava.eval_metrics.evaluate_metrics import calculate_exactmatch, calculate_f1score, bleu, calculate_appearance_with_normalization
from tabulate import tabulate

import warnings
warnings.simplefilter('ignore')

def parse_option():
    parser = argparse.ArgumentParser('Evaluation for LLaVA Generated Outputs', add_help=False)
    parser.add_argument('--gt', type=str, default="test.jsonl", help='path to ground truth file')
    parser.add_argument('--pred', type=str, default="answer-file-llava-zeorshot.jsonl", help='path to prediction file')
    args, unparsed = parser.parse_known_args()
    return args

def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as reader:
        for line in reader:
            data.append(json.loads(line))
    return data 

def align_dict_lists(gt, pred):
    gt_dict = {item['question_id']: item for item in gt}
    pred_dict = {item['question_id']: item for item in pred}

    common_ids = set(gt_dict.keys()) & set(pred_dict.keys())

    gt_aligned = [gt_dict[id] for id in common_ids]
    pred_aligned = [pred_dict[id] for id in common_ids]

    return gt_aligned, pred_aligned

def evaluate(gt, pred):    
    gt, pred = align_dict_lists(gt, pred)
    assert len(gt) == len(pred), "the length of gt is not the same as pred"
    
    scores = collections.defaultdict(list)
    closed_scores = collections.defaultdict(list)
    closed_questions_count = 0
    closed_questions_correct = 0
    
    for gt_item, pred_item in zip(gt, pred):
        gt_value = gt_item['answer'].lower()
        pred_value = pred_item['response'].lower()
        answer_type = gt_item['answer_type']

        # print(f"Processing question {gt_item['question_id']}: {answer_type}")
        
        if answer_type == 'OPEN':
            scores['exact_match'].append(calculate_exactmatch(pred_value, gt_value))
            f1_score, precision, recall = calculate_f1score(pred_value, gt_value)
            scores['f1'].append(f1_score)
            scores['precision'].append(precision)
            scores['recall'].append(recall)

            # Calculate BLEU scores with different weights
            weights = [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0)]
            bleu_scores = [sentence_bleu([gt_value.split()], pred_value.split())]
            for w in weights:
                bleu_score = sentence_bleu([gt_value.split()], pred_value.split(), 
                                         weights=w, 
                                         smoothing_function=SmoothingFunction().method1)
                bleu_scores.append(bleu_score)
            scores['bleu_scores'].append(bleu_scores)

        elif answer_type == 'CLOSED':
            f1_score, precision, recall = calculate_f1score(pred_value, gt_value)
            closed_scores['f1'].append(f1_score)
            closed_scores['precision'].append(precision)
            closed_scores['recall'].append(recall)
            closed_questions_count += 1
            if gt_value.lower() in pred_value.lower():
                closed_questions_correct += 1

    # Calculate average scores
    metrics = []
    
    if scores['exact_match']:  # 如果有开放式问题
        exact_match_avg = sum(scores['exact_match']) / len(scores['exact_match'])
        f1_score_avg = sum(scores['f1']) / len(scores['f1'])
        precision_avg = sum(scores['precision']) / len(scores['precision'])
        recall_avg = sum(scores['recall']) / len(scores['recall'])
        bleu_scores_avg = [sum(score_list) / len(score_list) for score_list in zip(*scores['bleu_scores'])]
        
        metrics.extend([
            ['Exact Match Score', exact_match_avg*100],
            ['Open F1 Score', f1_score_avg*100],
            ['Open Precision', precision_avg*100],
            ['Open Recall', recall_avg*100],
            ['BLEU Score', bleu_scores_avg[0]*100],
            ['BLEU Score (Weight 1)', bleu_scores_avg[1]*100],
            ['BLEU Score (Weight 2)', bleu_scores_avg[2]*100],
            ['BLEU Score (Weight 3)', bleu_scores_avg[3]*100],
        ])

    if closed_questions_count > 0:  # 如果有封闭式问题
        closed_score = closed_questions_correct / closed_questions_count
        closed_f1_score_avg = sum(closed_scores['f1']) / len(closed_scores['f1'])
        closed_precision_avg = sum(closed_scores['precision']) / len(closed_scores['precision'])
        closed_recall_avg = sum(closed_scores['recall']) / len(closed_scores['recall'])
        
        metrics.extend([
            ['Yes/No Accuracy', closed_score*100],
            ['Closed F1 Score', closed_f1_score_avg*100],
            ['Closed Precision', closed_precision_avg*100],
            ['Closed Recall', closed_recall_avg*100],
        ])

    results_table = tabulate(metrics, headers=['Metric', 'Performance (%)'])
    return results_table

if __name__ == '__main__':
    args = parse_option()
    gt = load_jsonl(args.gt)
    pred = load_jsonl(args.pred)
    results = evaluate(gt, pred)
    print(results)