# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

import argparse
import os
import pickle
from os.path import join

import pandas as pd
from tqdm import tqdm

from src.metrics import Metrics
from src.ranker import Ranker

config_parser = parser = argparse.ArgumentParser(description='', add_help=False)
parser.add_argument('--model_id', default=2, type=int, help='')

if __name__ == '__main__':
    args = vars(parser.parse_args())
    src_dir = './completions'
    out_dir = './datasets/llm_user/unannotated'
    k_vals = [100]

    model = 'mistralai_Mistral-7B-v0.1'
    config = 'best_of:5__top_n_tokens:5__temperature:1.0__top_p:1.0.pkl'
    model_completions = pickle.load(open(join(src_dir, model, config), 'rb'))
    assert len(model_completions) == 26394
    metrics = Metrics(saved_completions=model_completions)
    ranker = Ranker(saved_completions=model_completions)
    rank_fn = ranker.rank_fn

    examples = {}
    i = 0
    for c in tqdm(model_completions[:10]):
        for k in k_vals:
            prefix = c['dataset_row']['prefix']
            ground_truth = c['dataset_row']['gt_completion']
            completions, confidence = metrics.get_saved_completions(
                k, prefix, i, rank_fn, single_word=False, partial=True
            )
            if len(completions) > 0:
                accepted_text = metrics.exact_match_acceptance(
                    prefix, completions, prefix.strip() + ' ' + ground_truth.strip()
                )
            examples[i] = {
                'prefix': prefix,
                'turn_prefix': prefix.split('<|prompter|>')[-1],
                'ground_truth': ground_truth,
                'completions_sorted_by_ppl': completions,
                'accpeted_text': accepted_text.strip() if accepted_text is not None else '',
            }
        i += 1

    os.makedirs(join(out_dir, model), exist_ok=True)
    save_path = f'{out_dir}/{model}/{config}'

    # save examples as pkl file:
    f = open(f'{save_path}', 'wb')
    pickle.dump(examples, f)
    f.close()

    # save examples as csv:
    results_df = pd.DataFrame([e for e in examples.values()])
    results_df.to_csv(f'{save_path.replace(".pkl", ".csv")}', index=False)
