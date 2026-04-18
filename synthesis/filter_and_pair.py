import os
from datasets import Dataset

import json

import random

from prompt import *
import numpy as np

from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("--src_path", type=str, help="Path to the raw synthesized JSONL dataset")
parser.add_argument("--max_num_examples", type=int, default=100000, help="Maximum number of examples for the resulting dataset.")
parser.add_argument("--best_of_n", type=int, default=32, help="Number of candidate answers to keep per question.")

args = parser.parse_args()

related_pairing = False

samples = []

with open(args.src_path, "r") as src_json:
    if args.src_path.endswith("json"):
        samples.extend(json.load(src_json))
    elif args.src_path.endswith("jsonl"):
        samples.extend([json.loads(line) for line in src_json])

no_valid_q_or_r_count = 0

zero_std_count = 0
zero_valid_srs_count = 0

means = []
stds = []
max_scores = []
min_scores = []

new_samples = []

num_valid_srs = []
num_rubric_criteria = []

score_diffs = []

max_response_num_words = []
len_diff_ge_tau_count = 0
min_response_num_words = []

for s in samples:
    if not s["is_proposer_valid"] or not s["is_verifier_valid"]:
        no_valid_q_or_r_count = no_valid_q_or_r_count + 1
        continue

    valid_srs = [sr for sr in s["solver_responses"] if sr["is_verifier_scoring_valid"]]

    if len(valid_srs) == 0:
        zero_valid_srs_count = zero_valid_srs_count + 1
        continue

    valid_srs = valid_srs[:args.best_of_n]

    verifier_scores = [sr["verifier_scoring_score"] for sr in valid_srs]


    mean = np.mean(verifier_scores)
    std = np.std(verifier_scores)

    if std < 0.001:
        zero_std_count = zero_std_count + 1
        continue


    response_lengths = [len(valid_sr["solver_conversations"][-1]["content"].split(" ")) for valid_sr in valid_srs]
    candidates = []


    max_idx = np.argmax(verifier_scores)
    min_idx = np.argmin(verifier_scores)

    assert max_idx != min_idx, "{}".format(verifier_scores)

    max_response_words = valid_srs[max_idx]["solver_conversations"][-1]["content"].split(" ")
    min_response_words = valid_srs[min_idx]["solver_conversations"][-1]["content"].split(" ")
    
    if abs(len(max_response_words) - len(min_response_words)) > 100:
        len_diff_ge_tau_count = len_diff_ge_tau_count + 1
        continue

    means.append(mean)
    stds.append(std)
    
    max_scores.append(verifier_scores[max_idx])
    min_scores.append(verifier_scores[min_idx])

    num_valid_srs.append(len(valid_srs))
    num_rubric_criteria.append(len(s["verifier_rubric"]))

    score_diffs.append(verifier_scores[max_idx] - verifier_scores[min_idx])
    
    max_response_num_words.append(response_lengths[max_idx])
    min_response_num_words.append(response_lengths[min_idx])

    new_samples.append({
        "role": "solver",
        "chosen": valid_srs[max_idx]["solver_conversations"],
        "rejected": valid_srs[min_idx]["solver_conversations"],
        "chosen_score": verifier_scores[max_idx],
        "rejected_score": verifier_scores[min_idx],
        "ref_answer": s["proposer_ref_answer"]
    })


avg_mean = sum(means)/len(means) if len(means) > 0 else 0
avg_std = sum(stds)/len(stds) if len(stds) > 0 else 0
avg_max_scores = sum(max_scores)/len(max_scores) if len(max_scores) else 0
avg_min_scores = sum(min_scores)/len(min_scores) if len(min_scores) else 0
avg_num_valid_srs = sum(num_valid_srs)/len(num_valid_srs) if len(num_valid_srs) > 0 else 0
avg_num_rubric_criteria = sum(num_rubric_criteria)/len(num_rubric_criteria) if len(num_rubric_criteria) > 0 else 0
avg_score_diff = sum(score_diffs) / len(score_diffs) if len(score_diffs) > 0 else 0

print("{} examples have no valid question or rubric. {} examples have 0 std, {} examples have 0 valid srs. They are skipped.".format(no_valid_q_or_r_count,zero_std_count, zero_valid_srs_count))

print("avg_num_valid_srs={:.2f}; avg_num_rubric_critera={:.2f}; avg_mean={:.2f}; avg_std={:.2f}; avg_max_scores={:.2f}; avg_min_scores={:.2f}; avg_score_diff={:.2f}".format(avg_num_valid_srs, avg_num_rubric_criteria, avg_mean, avg_std, avg_max_scores, avg_min_scores, avg_score_diff))

avg_max_response_len = sum(max_response_num_words) / len(max_response_num_words)

avg_min_response_len = sum(min_response_num_words) / len(min_response_num_words)

print("avg max_response num words: {:.2f}; avg min_response num words: {:.2f}".format(avg_max_response_len, avg_min_response_len))
print("{} examples have chosen and rejected response differing by more than 100 words. They are skipped.".format(len_diff_ge_tau_count))

    

    

random.seed(42)
random.shuffle(new_samples)

new_samples = new_samples[:args.max_num_examples]

with open(args.src_path.replace(".json", "_dpo.json"), "w") as res_jsonl:
    for new_s in new_samples:
        res_jsonl.write(json.dumps(new_s) + "\n")