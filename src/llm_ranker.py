# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

import ast
import random

import numpy as np
import pandas as pd

from datasets import load_dataset
from inference import TGIRunner

hf_token_path = './resources/hf_token.txt'
random.seed(42)


class LlmRankerForPrism:
    def __init__(self):
        TGI_IMAGE = 'ghcr.io/huggingface/text-generation-inference:3.1.0'
        m1 = 'mistralai/Mistral-7B-Instruct-v0.3'
        args = {'model_id': m1, 'gpu_id': 0, 'best_of': 5, 'max_new_tokens': 20, 'top_n_tokens': 0, 'top_p': 1}

        name = args['model_id'].replace('/', '--')
        runner = TGIRunner(
            args['model_id'],
            args,
            TGI_IMAGE,
            f'./data/models--{name}',
            hf_token=open(hf_token_path, 'r').read(),
            gpu_id=0,
        )
        self.tgi_runner = runner
        self.survey_dataset = (
            load_dataset('HannahRoseKirk/prism-alignment', 'survey', revision='main')['train'].to_pandas().set_index('user_id')
        )

    def rank(self, prefix, completions, user_id, name):
        completions_lst = list(completions.items())
        for comp, tokens_dict in completions_lst:
            tokens_dict['log_likelihood'] = np.sum(tokens_dict['logprobs'])
            tokens_dict[name] = tokens_dict['log_likelihood']
        completions_lst = sorted(completions_lst, key=lambda x: x[1]['log_likelihood'], reverse=True)
        topk_completions = completions_lst[:5]
        topk_raw_completions = [x[0] for x in topk_completions]
        user_survey_info = self.survey_dataset.loc[user_id]
        prompt = self.create_mistral_prompt(user_survey_info, prefix, topk_raw_completions)
        try:
            llm_out = self.tgi_runner.run_inference(prompt)
            llm_order = [int(x) - 1 for x in llm_out.json()['generated_text'].split(', ')]
            for position, idx in enumerate(llm_order):
                completions_lst[idx][1][name] += 100 * (100 - position)
        except Exception:
            return

    def create_user_profile(self, user_id: str):
        user_profile = self.survey_dataset.loc[user_id]
        profile_lines = [
            f'- Age: {user_profile["age"]}',
            f'- Location: {user_profile["location"]["reside_country"]}',
            f'- Gender: {user_profile["gender"]}',
            f'- Education: {user_profile["education"]}',
            f'- Employment Status: {user_profile["employment_status"]}',
            f'- Self-description: "{user_profile["self_description"]}"',
            f'- Expectations from AI systems: "{user_profile["system_string"]}"',
        ]

        if any(user_profile['lm_usecases'].values()):
            profile_lines.append(
                '- Language model usage areas: '
                + ', '.join(
                    k.replace('_', ' ')
                    for k, v in user_profile['lm_usecases'].items()
                    if isinstance(v, (float, int)) and v > 0
                )
            )

        return chr(10).join(profile_lines)

    def create_mistral_prompt(self, user_profile: dict, prefix: str, suggestions: list[str]) -> str:
        suggestion_lines = '\n'.join([f'{i + 1}. {s}' for i, s in enumerate(suggestions)])
        n = len(suggestions)
        shuffle_order = ', '.join(str(i) for i in random.sample(range(1, n + 1), n))

        prompt = f"""<s>[INST]You are a helpful assistant personalizing autocomplete suggestions for a user.
The user has the following profile:

{self.create_user_profile(user_profile)}

The user typed the following prefix:
\"\"\"{prefix.strip()}\"\"\"

Here are the autocomplete suggestions:
{suggestion_lines}

Your task is to rank these suggestions from best to worst based on their *quality* and *usefulness to the user*, considering the user's background, goals, emotional state, and preferences.

Important guidelines:
- A suggestion does **not** need to complete the sentence fully. A short continuation (e.g., one or two words) can be better than a longer one *if it extends the prefix more naturally or effectively*.
- Prioritize suggestions that flow well with the prefix and align with the user’s context.
- Consider helpfulness, fluency, and relevance to user interests.

Return ONLY the ranked suggestion numbers, comma-separated (e.g., {shuffle_order}).
[/INST]"""
        return prompt


def extract_completions(completions_text):
    """Extract individual completions from the formatted text"""
    lines = completions_text.strip().split('\n')
    # Extract the text part after the number and period
    extracted = []
    import re

    for line in lines:
        match = re.match(r'^\d+\.\s*(.*)', line)
        if match:
            extracted.append(match.group(1))
    return extracted


def shuffle_completions(completions_list):
    """Create a shuffled version of completions with numbered formatting"""
    # Create a list of tuples containing (original_index, completion_text)
    indexed_completions = list(enumerate(completions_list, 1))
    # Shuffle the completions
    random.shuffle(indexed_completions)

    # Format shuffled completions with numbering
    formatted_shuffled = [f'{i}. {text}' for i, (_, text) in enumerate(indexed_completions, 1)]

    # Generate ranked_suggestions showing the new positions of original items
    # This creates a mapping from original position to new position
    ranking = {}
    for new_idx, (orig_idx, _) in enumerate(indexed_completions, 1):
        ranking[orig_idx] = new_idx

    # Convert ranking to comma-separated string in original order: "3,1,5,2,4"
    ranked_str = ','.join(str(ranking[i]) for i in range(1, len(completions_list) + 1))

    return '\n'.join(formatted_shuffled), ranked_str


def get_suggestions_rank(row):
    ranked_suggestions = [row['accepted_rank']]
    ranked_suggestions += [i for i in range(1, 6) if i != row['accepted_rank']]
    ranked_suggestions = ', '.join([str(i) for i in ranked_suggestions])
    return ranked_suggestions


def create_completions_df_for_claude():
    completions_df_dir_path = './results/saved_at_k/prism/word/mistral_rank_delete'
    ranker = LlmRankerForPrism()
    df = pd.read_csv(completions_df_dir_path + '/comps_df.csv', index_col=0)
    df['user_profile'] = df['user_id'].apply(ranker.create_user_profile)
    df['completions'] = df['completions'].apply(
        lambda s: '\n'.join(f'{i + 1}. {x.strip()}' for i, x in enumerate(ast.literal_eval(s)))
    )
    # split to train and test:
    # train set: first 10 users with accepted completions ranked #1
    train_uids = df[df['accepted_rank'] == 1].user_id.unique()[:10]
    train_df = df[df['user_id'].isin(train_uids)]
    train_df = train_df[train_df['accepted_rank'] == 1]
    new_columns = []
    for _, row in train_df.iterrows():
        completions_text = row['completions']
        completions_list = extract_completions(completions_text)

        shuffled_text, ranked_suggestions = shuffle_completions(completions_list)
        new_columns.append({'shuffled_completions': shuffled_text, 'ranked_suggestions': ranked_suggestions})
    train_df['shuffled_completions'] = [item['shuffled_completions'] for item in new_columns]
    train_df['ranked_suggestions'] = [item['ranked_suggestions'] for item in new_columns]

    train_df.to_csv(completions_df_dir_path + '/train.csv')

    # val set: first 50 users not in train
    val_uids = [uid for uid in df.user_id.unique() if uid not in train_uids][:50]
    val_df = df[df['user_id'].isin(val_uids)]
    val_df['ranked_suggestions'] = val_df.apply(get_suggestions_rank, axis=1)
    val_df.to_csv(completions_df_dir_path + '/val.csv')


if __name__ == '__main__':
    create_completions_df_for_claude()
