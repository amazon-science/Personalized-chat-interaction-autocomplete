# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

import argparse
import os
import pickle
import re
from os import listdir
from os.path import isfile, join

import pandas as pd
from tqdm import tqdm

from ranker import Ranker

config_parser = parser = argparse.ArgumentParser(description='', add_help=False)
parser.add_argument(
    '--model_id',
    default='mistralai_Mistral-7B-v0.1',
    help='model name as appears in huggingface or path to finetuned model',
    type=str,
)
parser.add_argument('--single_word', default=False, help='')
parser.add_argument('--partial', default=True, help='')
parser.add_argument('--char_prefixes', default=False, help='')
parser.add_argument('--on_reject_type', default='word', help='')
parser.add_argument('--dataset', default='dspy_val', help='')
parser.add_argument('--subset', default=None, help='')
parser.add_argument('--hparams', default=False, help='')
parser.add_argument(
    '--rank_by',
    default='personal_dspy_rank',
    choices=[
        'neg_perplexity',
        'log_likelihood',
        'max_logprob',
        'min_logprob',
        'mean_neg_entropy_norm',
        'mean_neg_entropy',
        'max_neg_entropy_norm',
        'max_neg_entropy',
        'min_neg_entropy_norm',
        'min_neg_entropy',
        'sequence_length',
        'random',
        'mlp_ranker',
        'mlp_reranker',
        'personal_cosine_mean',
        'personal_cosine_max',
        'personal_jaccard_mean',
        'personal_jaccard_max',
        'personal_edit_dist_mean',
        'personal_edit_dist_max',
        'personal_mistral_rank',
        'personal_dspy_rank',
    ],
)
parser.add_argument(
    '--mlp_path',
    default=None,
    help='',
)
parser.add_argument('--personal_w', default=0.0, help='')
parser.add_argument('--theta', default=None, help='')
parser.add_argument('--start_idx', type=int, default=0, help='')
parser.add_argument('--root_path', default='./', help='Root path for the project')
parser.add_argument(
    '--personalization_scheme',
    default='none',
    type=str,
    help="""(1) none - no personalization
                            (2) all_prev_convs - prepend all previous user conversations (without assistant replies)
                            (3) random_convs - randomly sample r previous user conversations (all turns, no assistant replies)
                            (4) random_turns - randomly sample r previous user turns (no assistant replies)
                            (5) recency_convs - r most recent previous user conversations (all turns, no assistant replies)
                            (6) recency_turns - r most recent previous user turns (no assistant replies)
                            """,
    choices=['none', 'all_prev_convs', 'random_convs', 'random_turns', 'recency_convs', 'recency_turns'],
)
parser.add_argument(
    '--personalization_r',
    default=None,
    type=int,
    help='How many previous interactions to use for personalization. Relevant only for personalization schemes that require r.',
)


# Compile regex patterns once for better performance
_SPLIT_PUNC_PATTERN = re.compile(r'[^ \s,.!?()]+|[?!.,()]')
_PUNCTUATION_PATTERN = re.compile(r'[^\w\s]')
COMPLETIONS_DF = {}
COLLECT_COMPLETIONS_DF = True


def split_punc(text):
    return _SPLIT_PUNC_PATTERN.findall(text.lower())


class Metrics:
    def __init__(self, saved_completions):
        self.saved_completions = saved_completions
        self.comp_dict = {}
        for completion in self.saved_completions:
            self.comp_dict[completion['dataset_row']['prefix'].lower()] = completion

    def saved_typing_at_k(
        self,
        k,
        prefix,
        ground_truth,
        completions_fn,
        acceptance_fn,
        ranker,
        args,
        len_fn='chars',
        prefix_idx=None,
        char_prefixes=False,
        user_id=None,
    ):
        if len_fn == 'chars':
            self.len_fn = len
        elif len_fn == 'words':
            self.len_fn = lambda x: len(x.split())  # noqa: E731

        if char_prefixes:
            full_ground_truth = prefix + ground_truth
        else:
            full_ground_truth = prefix.strip() + ' ' + ground_truth.strip()

        ground_truth_words = split_punc(full_ground_truth)
        prefix_words = split_punc(prefix)
        len_to_complete = self.len_fn(full_ground_truth) - self.len_fn(prefix.strip())

        num_accepts, accepted_length, num_suggested, num_abstained = 0, 0, 0, 0
        (
            accepted_text_indices,
            accepted_text_strs,
            accepted_ranks,
            accepted_confidence,
        ) = [], [], [], []

        while self.len_fn(prefix) < self.len_fn(full_ground_truth) and prefix_words != ground_truth_words:
            completions, confidence = completions_fn(
                k,
                prefix,
                prefix_idx,
                ranker,
                single_word=args['single_word'],
                partial=args['partial'],
                char_prefixes=args['char_prefixes'],
                user_id=user_id,
            )

            # abstain if not confident
            if args['theta'] is not None:
                k_new = len([c for c in confidence if c >= args['theta']])
                completions, confidence = completions[:k_new], confidence[:k_new]
                if k_new == 0:
                    num_abstained += 1

            if len(completions) > 0:
                num_suggested += 1
                accepted_text = acceptance_fn(
                    prefix, completions, full_ground_truth, args.get('add_leading_space', False)
                )
                if accepted_text is not None and accepted_text.strip() != '':
                    num_accepts += 1
                    accepted_length += self.len_fn(accepted_text)
                    accepted_text_indices.append((len(prefix), len(prefix + accepted_text) + 1))
                    if COLLECT_COMPLETIONS_DF:
                        COMPLETIONS_DF[prefix.lower()] = {
                            'completions': completions,
                            'accepted_rank': completions.index(accepted_text),
                            'full_gt': full_ground_truth,
                            'user_id': user_id,
                        }
                    if char_prefixes:
                        prefix = prefix + accepted_text
                        accepted_text_strs.append(accepted_text)
                    else:
                        prefix = prefix.strip() + ' ' + accepted_text.strip()
                        accepted_text_strs.append(accepted_text)
                    rank = completions.index(accepted_text)
                    accepted_ranks.append(rank + 1)
                    accepted_confidence.append(confidence[rank])
                    prefix_words = split_punc(prefix)
                    continue

            # if no completions were suggested, or none of the completions is accepted, simulate the user typing another word or char
            prefix_words = split_punc(prefix)
            if args['on_reject_type'] == 'char':
                prefix += full_ground_truth[len(prefix)]
            else:
                if prefix_words == ground_truth_words:
                    break
                if full_ground_truth[len(prefix)] == ' ':
                    prefix = prefix.strip() + ' ' + ground_truth_words[len(prefix_words)]
                else:
                    prefix = prefix + ground_truth_words[len(prefix_words)]

            prefix_words = split_punc(prefix)

        min_num_clicks = len_to_complete - 1 if (len_to_complete - 1) > 0 else 1
        accepted_length_words = sum([len(t.split()) for t in accepted_text_strs])

        res = {
            f'saved@{k}': (accepted_length - num_accepts) / min_num_clicks,
            f'num_accepts@{k}': num_accepts,
            f'accepted_length_chars@{k}': accepted_length,
            f'accepted_length_words@{k}': accepted_length_words,
            f'accepted_text_indices@{k}': accepted_text_indices,
            f'accepted_text_strs@{k}': accepted_text_strs,
            f'acceptance_rate@{k}': num_accepts / num_suggested if num_suggested > 0 else 0,
            f'completed_length@{k}': accepted_length / len_to_complete,
            f'chars_per_accept@{k}': accepted_length / num_accepts if num_accepts > 0 else 0,
            f'words_per_accept@{k}': accepted_length_words / num_accepts if num_accepts > 0 else 0,
            f'accepted_ranks@{k}': accepted_ranks,
            f'mean_accepted_ranks@{k}': sum(accepted_ranks) / len(accepted_ranks) if len(accepted_ranks) > 0 else 0,
            f'mean_accepted_confidence@{k}': sum(accepted_confidence) / len(accepted_confidence)
            if len(accepted_confidence) > 0
            else 0,
            f'accepted_confidence@{k}': accepted_confidence,
            f'num_abstained@{k}': num_abstained,
            f'num_suggested@{k}': num_suggested,
            f'abstain_ratio@{k}': num_abstained / (num_abstained + num_suggested)
            if (num_abstained + num_suggested) > 0
            else 0,
        }
        return res

    def exact_match_acceptance(self, prefix, completions, full_ground_truth, add_leading_space=False):
        completions = sorted([c for c in completions if c.strip() != ''], key=len, reverse=True)
        for completion in completions:
            if add_leading_space:
                suggested_text = prefix + ' ' + completion
            else:
                suggested_text = prefix + completion
            if len(suggested_text) > len(full_ground_truth):
                continue

            suggested_text_words = split_punc(suggested_text)
            ground_truth_words = split_punc(full_ground_truth)
            if (
                suggested_text_words == ground_truth_words[: len(suggested_text_words)]
                and suggested_text.lower() == full_ground_truth[: len(suggested_text)].lower()
            ):
                return completion

        return None

    def get_saved_completions(
        self, k, prefix, prefix_idx, ranker, partial=True, single_word=False, char_prefixes=False, user_id=None
    ):
        try:
            saved_completion = self.comp_dict[prefix.lower()]
            key = saved_completion['completion']['generated_text']
        except Exception:
            # print(f"missing completion, idx {prefix_idx}")
            return [], []
        completions = {
            key: {
                'tokens': [t['text'] for t in saved_completion['completion']['details']['tokens']],
                'logprobs': [t['logprob'] for t in saved_completion['completion']['details']['tokens']],
                'top_tokens_logprobs': [
                    [t1['logprob'] for t1 in t2] for t2 in saved_completion['completion']['details']['top_tokens']
                ]
                if 'top_tokens' in saved_completion['completion']['details'].keys()
                else [],
            }
        }
        if 'best_of_sequences' in saved_completion['completion']['details'].keys():
            for c in saved_completion['completion']['details']['best_of_sequences']:
                completions[c['generated_text']] = {
                    'tokens': [t['text'] for t in c['tokens']],
                    'logprobs': [t['logprob'] for t in c['tokens']],
                    'top_tokens_logprobs': [[t1['logprob'] for t1 in t2] for t2 in c['top_tokens']]
                    if 'top_tokens' in saved_completion['completion']['details'].keys()
                    else [],
                }

        # include all partial completions
        if partial:
            partial_completions = {}
            for _, tokens_dict in completions.items():
                for j in range(1, len(tokens_dict['tokens'])):
                    # if the next token is a new word or is the end of the sequence
                    if tokens_dict['tokens'][j].startswith(' ') or bool(
                        _PUNCTUATION_PATTERN.fullmatch(tokens_dict['tokens'][j])
                    ):
                        partial_completions[''.join(tokens_dict['tokens'][:j])] = {
                            'tokens': tokens_dict['tokens'][:j],
                            'logprobs': tokens_dict['logprobs'][:j],
                            'top_tokens_logprobs': tokens_dict['top_tokens_logprobs'][:j],
                        }
            completions.update(partial_completions)

        if single_word:
            single_word_completions = {}
            for _, tokens_dict in completions.items():
                for j in range(1, len(tokens_dict['tokens'])):
                    # if the next token is a new word or is the end of the sequence
                    if tokens_dict['tokens'][j].startswith(' ') or bool(
                        _PUNCTUATION_PATTERN.fullmatch(tokens_dict['tokens'][j])
                    ):
                        single_word_completions[''.join(tokens_dict['tokens'][:j])] = {
                            'tokens': tokens_dict['tokens'][:j],
                            'logprobs': tokens_dict['logprobs'][:j],
                            'top_tokens_logprobs': tokens_dict['top_tokens_logprobs'][:j],
                        }
                        break
                if len(''.join(tokens_dict['tokens']).split()) == 1:
                    single_word_completions[''.join(tokens_dict['tokens'])] = {
                        'tokens': tokens_dict['tokens'],
                        'logprobs': tokens_dict['logprobs'],
                        'top_tokens_logprobs': tokens_dict['top_tokens_logprobs'],
                    }

            completions = single_word_completions

        # manually remove \n from completions as it's never part of the ground truth completion
        to_delete = set()
        to_add = []
        for c, tokens_dict in completions.items():
            if c.endswith('\n'):
                to_delete.add(c)
                c = c.replace('\n', '')
                tokens_dict['tokens'] = tokens_dict['tokens'][:-1]
                tokens_dict['logprobs'] = tokens_dict['logprobs'][:-1]
                to_add.append((c, tokens_dict))
            elif c.strip() == '':
                to_delete.add(c)
        for c in to_delete:
            del completions[c]
        for key, val in to_add:
            completions[key] = val

        if '' in completions.keys():
            del completions['']

        # sort candidates
        sorted_completions, confidence = ranker.rank(prefix, completions, user_id)

        # return top k completions and confidence
        return sorted_completions[:k], confidence[:k]


models_list = [
    'microsoft_Phi-3.5-mini-instruct',
    'mistralai/Mistral-7B-v0.1',
    'HuggingFaceH4/zephyr-7b-beta',
    'meta-llama/Meta-Llama-3.1-8B',
    'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'google/gemma-7b',
    'google/gemma-7b-it',
    'microsoft/Phi-3-mini-4k-instruct',
    'openai-community/gpt2-xl',
    'HuggingFaceTB/SmolLM-1.7B',
    'state-spaces/mamba-2.8b',
    'xiuyul/mamba-2.8b-zephyr',
]


def get_src_out_dirs(args):
    root_path = args['root_path']

    # For dspy_val dataset, use prism completions but filter later
    if args['dataset'] == 'dspy_val':
        src_dir = os.path.join(root_path, 'completions/prism')
        out_dir = os.path.join(root_path, 'results/saved_at_k/dspy_val')
    else:
        src_dir = os.path.join(root_path, f'completions/{args["dataset"]}')
        out_dir = os.path.join(root_path, f'results/saved_at_k/{args["dataset"]}')

    if args['single_word']:
        out_dir += '/baselines/single_word'
    if args['char_prefixes']:
        if args['dataset'] == 'oasst':
            src_dir += '/char/full_149K'
            out_dir += '/char/full_149K'
        elif args['dataset'] == 'sharegpt':
            src_dir += '/char/sample_144K'
            out_dir += '/char/sample_144K'
        elif args['dataset'] == 'dspy_val':
            src_dir += '/char'
            out_dir += '/char'
        if args['on_reject_type'] == 'word':
            out_dir += '_on_reject_type_word'
    else:
        if args['dataset'] == 'oasst':
            src_dir += '/word/val'
            out_dir += '/word/val'
        elif args['dataset'] == 'sharegpt':
            src_dir += '/word/sample_22K'
            out_dir += '/word/sample_22K'
        elif 'prism' in args['dataset'] or args['dataset'] == 'dspy_val':
            src_dir += '/word'
            out_dir += '/word'
        elif args['dataset'] == 'wildchat':
            src_dir += '/word'
            out_dir += '/word'

    if args['subset'] is not None:
        src_dir += f'/subset_{args["subset"]}'
        out_dir += f'/subset_{args["subset"]}'

    if args['hparams']:
        src_dir = './completions/oasst/word_subset_0.25'
        out_dir = './results/saved_at_k/oasst_subset_0.25'
    return src_dir, out_dir


def filter_completions_for_dspy_val(model_completions, root_path):
    """Filter completions to only include those with user_ids in dspy/val.csv"""
    dspy_val_path = os.path.join(root_path, 'dspy/val.csv')
    dspy_val_df = pd.read_csv(dspy_val_path)
    dspy_val_user_ids = set(dspy_val_df['user_id'])

    filtered_completions = []
    for completion in model_completions:
        user_id = completion['dataset_row']['user_id']
        if user_id in dspy_val_user_ids:
            filtered_completions.append(completion)

    return filtered_completions


def get_k_vals(args):
    k_vals = [1, 2, 3, 5, 100]
    if 'personal' in args['rank_by']:
        k_vals = [1, 2, 3, 5]
    # k_vals = [1, 2, 3, 5, 10, 20, 50, 100]
    if args['single_word']:
        k_vals = [1, 2, 3, 4, 5]
    if args['hparams']:
        k_vals = [100]
    return k_vals


if __name__ == '__main__':
    args = vars(parser.parse_args())
    args['theta'] = float(args['theta']) if args['theta'] is not None else None
    print(args)
    src_dir, out_dir = get_src_out_dirs(args)
    k_vals = get_k_vals(args)
    if args['model_id'] is not None:
        models_list = [args['model_id']]

    for model in models_list:
        if 'mamba' in model and 'entropy' in args['rank_by']:
            continue
        model = model.replace('/', '_')
        model_dir = join(src_dir, model)
        os.makedirs(join(out_dir, model), exist_ok=True)

        # Filter files based on personalization scheme
        if args['personalization_scheme'] == 'none':
            # For no personalization, look for files that don't have personalization patterns
            def personalization_filter(f):
                return not any(
                    pattern in f
                    for pattern in [
                        'all_prev_convs',
                        'random_convs',
                        'random_turns',
                        'recency_convs',
                        'recency_turns',
                        'intro',
                    ]
                )

        else:
            # For specific personalization schemes, look for files with that pattern
            scheme = args['personalization_scheme']
            if scheme in ['recency_convs', 'recency_turns']:
                # For recency-based schemes, also include the r parameter in the filter
                r = args['personalization_r']

                def personalization_filter(f):
                    return f'{scheme}' in f and f'personalization_r:{r}' in f

            else:

                def personalization_filter(f):
                    return scheme in f

        files = sorted([
            f
            for f in listdir(model_dir)
            if isfile(join(model_dir, f)) and ('best_of:5' in f) and personalization_filter(f)
        ])

        for f_i, file in enumerate(files[args['start_idx'] :]):
            output_path = f'{out_dir}/{model}/{file.replace(".pkl", "")}__rank_by:{args["rank_by"]}__personal_w:{args["personal_w"]}.csv'
            model_completions = []
            print(f'File index: {f_i + args["start_idx"]}, Model: {model}, config: {file}')
            with open(join(model_dir, file), 'rb') as cf:
                model_completions = pickle.load(cf)
            assert len(model_completions) > 0
            print(f'found completions, length: {len(model_completions)}')

            # Filter completions for dspy_val dataset
            if args['dataset'] == 'dspy_val':
                model_completions = filter_completions_for_dspy_val(model_completions, args['root_path'])
                print(f'Filtered to dspy_val subset, length: {len(model_completions)}')
            elif args['dataset'] == 'prism' and len(model_completions) < 2825:
                print('Not enough completions. Skipping.')
                continue
            results_df = None
            if isfile(output_path):
                results_df = pd.read_csv(output_path, index_col=0).reset_index()
                if len(results_df) == len(model_completions):
                    print('exists, skipping file')
                    continue
                else:
                    model_completions = model_completions[len(results_df) :]

            print(f'Running metrics, results output file: {output_path}')
            metrics = Metrics(saved_completions=model_completions)
            # For dspy_val, use prism as the dataset for ranker initialization
            # since dspy_val uses prism completions under the hood
            ranker_dataset = 'prism' if args['dataset'] == 'dspy_val' else args['dataset']
            ranker = Ranker(
                saved_completions=model_completions,
                confidence_measure=args['rank_by'],
                model_name=args['model_id'],
                mlp_path=args['mlp_path'],
                personal_w=args['personal_w'],
                dataset=ranker_dataset,
                root_path=args['root_path'],
            )

            rows = []
            for i, c in enumerate(tqdm(model_completions)):
                rows.append(c['dataset_row'].copy())
                for k in k_vals:
                    prefix = c['dataset_row']['prefix']
                    ground_truth = c['dataset_row']['gt_completion']

                    if (
                        args['on_reject_type'] == 'word'
                        and args['char_prefixes']
                        and not ground_truth.startswith(' ')
                        and not ground_truth.startswith('\n')
                    ):
                        continue

                    saved_typing_at_k_results = metrics.saved_typing_at_k(
                        k,
                        prefix,
                        ground_truth,
                        completions_fn=metrics.get_saved_completions,
                        acceptance_fn=metrics.exact_match_acceptance,
                        ranker=ranker,
                        args=args,
                        prefix_idx=i,
                        char_prefixes=args['char_prefixes'],
                        user_id=c['dataset_row']['user_id'],
                    )
                    rows[-1].update(saved_typing_at_k_results)

                if i > 0 and ((i % 100 == 0) or ('dspy' in args['rank_by'] and i % 5 == 0)):
                    if results_df is not None:
                        results_df = pd.concat([results_df, pd.DataFrame(rows)])
                    else:
                        results_df = pd.DataFrame(rows)
                    results_df.to_csv(output_path, index=False)
                    rows = []

            results_df = pd.concat([results_df, pd.DataFrame(rows)])
            results_df.to_csv(output_path, index=False)

            if COLLECT_COMPLETIONS_DF:
                comps_df_rows = [
                    {
                        'prefix': prefix.replace('<|prompter|> ', ''),
                        'completions': c['completions'],
                        'accepted_rank': c['accepted_rank'] + 1,
                        'user_id': c['user_id'],
                        'full_gt': c['full_gt'].replace('<|prompter|> ', ''),
                    }
                    for prefix, c in COMPLETIONS_DF.items()
                ]
                pd.DataFrame(comps_df_rows).to_csv('/'.join(output_path.split('/')[:-1] + ['comps_df.csv']))
