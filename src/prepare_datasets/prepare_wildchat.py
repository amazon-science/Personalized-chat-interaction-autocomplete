# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

import ast
import datetime
import os
import random
import re

import numpy as np
import pandas as pd
from Levenshtein import distance as levenshtein_distance
from tqdm.notebook import tqdm

from datasets import load_dataset

tqdm.pandas()
np.random.seed(42)
random.seed(42)
CHAR_LIMIT = 25000


def parse_conversation(conversation):
    # parse conversation
    data_str = conversation
    data_str = data_str.replace('datetime.datetime', '')
    data_str = data_str.replace('tzinfo=<UTC>', "'+00:00'")  # Replace tzinfo placeholder
    parsed_data = ast.literal_eval(data_str)
    # Convert timestamp strings to actual datetime objects
    for item in parsed_data:
        if isinstance(item.get('timestamp'), tuple):
            if len(item['timestamp']) == 7:
                year, month, day, hour, minute, second, tz = item['timestamp']
            elif len(item['timestamp']) == 6:
                year, month, day, hour, minute, tz = item['timestamp']
                second = 0  # Default second to 0
            # Create a datetime object and apply timezone info
            dt_obj = datetime.datetime(year, month, day, hour, minute, second)
            item['timestamp'] = dt_obj.strftime('%Y-%m-%d %H:%M:%S') + tz  # Format as required
    return parsed_data


def is_conv_eng(conversation):
    for message in conversation:
        if message['language'] != 'English':
            return False
        # Check percentage of non-English Unicode characters
        non_eng_chars = sum(1 for c in message['content'] if not c.isascii())
        if non_eng_chars / len(message['content']) > 0.05:
            return False
    return True


def is_conv_not_long(conversation):
    for message in conversation:
        if message['role'] == 'user' and len(message['content']) > 2000:
            return False
    return True


# filter non-english conversations and very long conversations
def filter_conversations(conv):
    if isinstance(conv, str):
        conversation = parse_conversation(conv)
    else:
        conversation = conv
    return is_conv_eng(conversation) and is_conv_not_long(conversation)


def get_prompt(conversation):
    history = ''
    for message in conversation[:-1]:
        role = 'prompter' if message['role'] == 'user' else 'assistant'
        text = message['content']
        history += f'<|{role}|> {text}\n'
    return history


def build_user_prompts(dataset):
    print('building user prompts')
    user_prompts = []
    for index, row in dataset.iterrows():
        # if index % 1000 == 0:
        #     print(f"{index}/{len(dataset)}")
        conv = row['conversation']
        if isinstance(conv, str):
            conversation = parse_conversation(conv)
        else:
            conversation = conv

        for i, message in enumerate(conversation):
            if message['role'] == 'user':
                history = get_prompt(conversation[: i + 1])
                user_prompts.append(
                    dict(
                        user_id=row['hashed_ip'],
                        conv_id=row['conversation_hash'],
                        timestamp=conversation[i + 1]['timestamp'],
                        turn=i // 2 + 1,
                        text=message['content'],
                        history=history,
                    )
                )
    return pd.DataFrame(user_prompts)


def max_len(conv):
    conversation = parse_conversation(conv)
    return max([len(message['content']) for message in conversation if message['role'] == 'user'])


def compute_edit_distance_similarity(text1, text2):
    """
    Computes the similarity based on Levenshtein edit distance.
    Returns a similarity score between 0 and 1 (1 means identical).
    """
    max_len = max(len(text1), len(text2))
    if max_len == 0:
        return 1  # If both strings are empty, consider them identical

    edit_dist = levenshtein_distance(text1, text2)
    similarity = 1 - (edit_dist / max_len)  # Normalize to [0,1]

    return similarity


def compute_jaccard_similarity(text1, text2):
    """
    Computes the similarity based on Jaccard distance.
    Returns a similarity score between 0 and 1 (1 means identical).
    """
    set_1, set_2 = set(text1.split()), set(text2.split())  # Tokenize into words
    intersection = len(set_1 & set_2)
    union = len(set_1 | set_2)
    return intersection / union if union != 0 else 0  # Avoid division by zero


def remove_similar_older_rows(df, threshold):
    """
    Removes only the older row in each 'user_id' group when two texts exceed 'threshold' similarity.
    """

    def filter_group(group):
        """
        Within each group, remove the earlier timestamped row if two texts are too similar.
        """
        # Sort group by timestamp (oldest to newest)
        group = group.sort_values(by='timestamp')

        # Track conversations to remove
        to_remove = set()

        # Compare each row with later rows
        texts = group['text'].tolist()
        indices = group.conv_id.tolist()

        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):  # Only compare later rows to earlier ones
                if indices[i] == indices[j]:
                    continue
                similarity = compute_edit_distance_similarity(texts[i], texts[j])

                if similarity > threshold:
                    to_remove.add(indices[i])  # Remove earlier row (i)

        return group.loc[~group.conv_id.isin(to_remove)]  # Keep non-removed conversations

    # Apply filtering to each group
    return df.groupby('user_id', group_keys=False).apply(filter_group)


def build_word_prefixes(user_prompts):
    user_prompts_df_w_prefix = pd.DataFrame([
        {
            **d,
            'prefix': d['history'] + '<|prompter|> ' + ' '.join(d['text'].split()[:i]),
            'prefix_k': i,
            'gt_completion': re.search(r'^.*?[.!?\n]+|.*$', ' '.join(d['text'].split()[i:]), flags=re.DOTALL)
            .group()
            .strip(),
            # gt is only until end-of-line
        }
        for _, d in user_prompts.iterrows()
        for i in range(len(d['text'].split()))
    ])

    return user_prompts_df_w_prefix


def build_prefixes(user_prompts):
    user_prompts_df_w_prefix = build_word_prefixes(user_prompts)

    # remove rows where there is no history and prefix is empty
    user_prompts_df_w_prefix = user_prompts_df_w_prefix[
        user_prompts_df_w_prefix['prefix'] != '<|prompter|> '
    ].reset_index(drop=True)

    return user_prompts_df_w_prefix


def prepare_base_dataset(save_path, conv_sim_threshold, train=False, num_samples=None):
    filtered_dataset_path = './datasets/wildchat/dataset_eng_all.csv'
    os.makedirs('./datasets/wildchat', exist_ok=True)
    if not os.path.isfile(filtered_dataset_path):
        # load dataset
        dataset = load_dataset('allenai/WildChat-1M', revision='main')['train']  # the dataset only contains train, we split it later
        if num_samples is not None:
            dataset = dataset.select(range(min(num_samples, len(dataset))))
        filtered_dataset = dataset.filter(lambda entry: entry['conversation'][0]['language'] == 'English')
        filtered_dataset = pd.DataFrame(filtered_dataset)

        # refilter non-english and too long conversations
        filtered_dataset['filter'] = filtered_dataset['conversation'].apply(filter_conversations)
        filtered_dataset = filtered_dataset[filtered_dataset['filter']]
        filtered_dataset = filtered_dataset.drop(['filter'], axis=1)

        # remove conversations with the same id that appear for multiple users
        duplicate_conv_ids = filtered_dataset.groupby('conversation_hash')['hashed_ip'].nunique()
        conflicting_conv_ids = duplicate_conv_ids[duplicate_conv_ids > 1].index
        filtered_dataset = filtered_dataset[
            ~filtered_dataset['conversation_hash'].isin(conflicting_conv_ids)
        ].reset_index(drop=True)

        # remove users with less than 6 conversations and more than 1000
        user_conv_count = filtered_dataset['hashed_ip'].value_counts()
        user_conv_count = user_conv_count[(user_conv_count >= 6) & (user_conv_count <= 1000)]
        filtered_dataset = filtered_dataset[filtered_dataset.hashed_ip.isin(user_conv_count.index)]
        filtered_dataset = filtered_dataset.reset_index(drop=True)
        filtered_dataset.to_csv(filtered_dataset_path, index=False)
    else:
        filtered_dataset = pd.read_csv(filtered_dataset_path)

    # sample val set users: 20% of the users
    user_conv_count = filtered_dataset['hashed_ip'].value_counts()
    n_val_users = len(user_conv_count) // 5
    val_user_sample = user_conv_count.sort_index().iloc[:n_val_users]
    val_set = filtered_dataset[filtered_dataset.hashed_ip.isin(val_user_sample.index)].reset_index(drop=True)

    # build prompts
    val_user_prompts = build_user_prompts(val_set)
    print(f'Num prompts before removing rephrases: {len(val_user_prompts)}')
    # remove rephrased messages
    val_prompts_filtered = remove_similar_older_rows(val_user_prompts, conv_sim_threshold)
    print(
        f'Num prompts after removing rephrases: {len(val_prompts_filtered)}. Ratio of Rephrased prompts: {1 - len(val_prompts_filtered) / len(val_user_prompts)}'
    )

    # refilter users left with less than 6 conversations after rephrase cleaning
    user_conv_count = val_prompts_filtered.groupby('user_id')['conv_id'].nunique()
    user_conv_count = user_conv_count[user_conv_count >= 6]
    val_prompts_filtered = val_prompts_filtered[val_prompts_filtered.user_id.isin(user_conv_count.index)].reset_index(
        drop=True
    )

    filtered_dfs = []

    # for each user, sample a conversation that has at least 5 preceding conversations and drop the later ones
    # this is done to downsample the data and make sure we don't bias towards the last recorded message
    for i, user_df in val_prompts_filtered.groupby('user_id'):
        user_df = user_df.sort_values(by='timestamp')
        conv_ids = user_df['conv_id'].drop_duplicates().tolist()
        if len(conv_ids) == 6:
            n = 6
        else:
            n = np.random.randint(6, len(conv_ids))
        conv_ids_left = conv_ids[:n]
        user_df = user_df[user_df['conv_id'].isin(conv_ids_left)]
        filtered_dfs.append(user_df)

    sampled_val_prompts = pd.concat(filtered_dfs).reset_index(drop=True)
    user_prompts_w_prefix = build_prefixes(sampled_val_prompts)

    # save to file
    user_prompts_w_prefix.to_csv(save_path, index=False)


def get_user_prev_convs(prev_convs_df):
    prev_convs = (
        prev_convs_df.groupby(['user_id', 'conv_id'])
        .apply(
            lambda x: {
                'latest_timestamp': x['timestamp'].max(),
                'interactions': x.sort_values(by='timestamp')['text'].drop_duplicates().tolist(),
            }
        )
        .reset_index(name='conv_data')
    )

    # Nest the dictionaries by user_id
    prev_convs = (
        prev_convs.groupby('user_id')
        .apply(lambda x: {row['conv_id']: row['conv_data'] for _, row in x.iterrows()})
        .to_dict()
    )

    # Sort by latest timestamp
    prev_convs = {
        user_id: {
            conv_id: conv_data
            for conv_id, conv_data in sorted(conv_data.items(), key=lambda item: item[1]['latest_timestamp'])
        }
        for user_id, conv_data in prev_convs.items()
    }

    return prev_convs


def concatenate_convs_with_limit(prev_convs):
    total_length = 0
    used_convs = []
    used_conv_ids = []
    conv_delimiter_template = '## Conversation {n} ##\n'

    # concatenate interactions until the character limit is reached
    for conv_id, conv_info in reversed(prev_convs.items()):
        all_conv_interactions = '<|prompter|> ' + '\n<|prompter|> '.join(conv_info['interactions']) + '\n\n'
        interactions_length = len(all_conv_interactions + conv_delimiter_template) + 4

        if total_length + interactions_length <= CHAR_LIMIT:
            used_convs.append(all_conv_interactions)
            total_length += interactions_length
            used_conv_ids.append(conv_id)
        else:
            # Stop adding prompts once the limit is exceeded
            break

    combined_convs = (
        ''.join([conv_delimiter_template.format(n=str(i + 1)) + conv for i, conv in enumerate(used_convs[::-1])])
        + '## Current Conversation ##\n'
    )
    return combined_convs, used_conv_ids


def concatenate_turns_with_limit(user_prev_turns_df):
    user_prev_turns = [(conv_id, text) for conv_id, text in zip(user_prev_turns_df.conv_id, user_prev_turns_df.text)]
    total_length = 0
    used_turns = []
    used_conv_ids = []
    turn_delimiter_template = '## Message {n} ##\n'

    # concatenate interactions until the character limit is reached
    for conv_id, text in reversed(user_prev_turns):
        full_interaction = '<|prompter|> ' + text + '\n\n'
        interactions_length = len(full_interaction + turn_delimiter_template) + 2

        if total_length + interactions_length <= CHAR_LIMIT:
            used_turns.append(full_interaction)
            total_length += interactions_length
            used_conv_ids.append(conv_id)
        else:
            # Stop adding prompts once the limit is exceeded
            break

    combined_turns = (
        ''.join([turn_delimiter_template.format(n=str(i + 1)) + message for i, message in enumerate(used_turns[::-1])])
        + '## Current Message ##\n'
    )
    return combined_turns, used_conv_ids, len(used_turns)


def combine_previous_interactions(prev_interactions, interaction_type):
    combined_interactions = {}
    used_conv_ids = {}

    if interaction_type == 'conv':
        for user_id, convs in prev_interactions.items():
            # Use the new function to get concatenated prompts and the used conv_ids
            user_combined_convs, user_used_conv_ids = concatenate_convs_with_limit(convs)
            combined_interactions[user_id] = user_combined_convs
            used_conv_ids[user_id] = user_used_conv_ids

    elif interaction_type == 'turn':
        total_turn_count = 0
        for user_id in prev_interactions.user_id.unique():
            user_combined_turns, user_used_conv_ids, used_turn_count = concatenate_turns_with_limit(
                prev_interactions[prev_interactions.user_id == user_id]
            )
            total_turn_count += used_turn_count
            combined_interactions[user_id] = user_combined_turns
            used_conv_ids[user_id] = user_used_conv_ids

    return combined_interactions, used_conv_ids


def personalize_dataset(dataset, personalization_scheme, personalization_r, personalization_pool_size=None):
    # get id of the latest conversation of each user
    dataset = dataset.sort_values(by=['user_id', 'timestamp']).reset_index(drop=True)
    latest_conv_ids = dataset.groupby('user_id', as_index=False)['conv_id'].last()['conv_id']
    if personalization_scheme == 'none':
        # take the first turn of the last conversation
        return dataset[dataset['conv_id'].isin(latest_conv_ids) & (dataset['turn'] == 1)].reset_index(drop=True)
    else:
        # get all previous prompts for each user
        latest_interactions = dataset[dataset['conv_id'].isin(latest_conv_ids)].reset_index(drop=True)
        prev_convs_df = dataset[~dataset['conv_id'].isin(latest_conv_ids)]
        latest_interactions.loc[:, 'prefix_before'] = latest_interactions['prefix']
        prev_conv_ids = prev_convs_df.groupby('user_id')['conv_id'].unique().to_dict()
        latest_interactions.loc[:, 'all_prev_conv_ids'] = latest_interactions['user_id'].map(prev_conv_ids)
        latest_interactions.loc[:, 'all_prev_conv_count'] = latest_interactions['all_prev_conv_ids'].apply(
            lambda x: len(x)
        )

        if 'convs' in personalization_scheme:
            user_prev_convs = get_user_prev_convs(prev_convs_df)
            # all previous user prompts (without assistant replies)
            if personalization_scheme == 'all_prev_convs':
                prev_convs = user_prev_convs
            # randomly sample r previous conversations
            elif 'random' in personalization_scheme:
                prev_convs = {}
                for user_id, convs in user_prev_convs.items():
                    conv_ids = [conv_id for conv_id in convs.keys()]
                    r_conv_ids = random.sample(conv_ids, min(personalization_r, len(conv_ids)))
                    prev_convs[user_id] = {
                        conv_id: conv_data for conv_id, conv_data in convs.items() if conv_id in r_conv_ids
                    }
            # take prompts from r most recent conversations
            elif 'recency' in personalization_scheme:
                prev_convs = {}
                for user_id, convs in user_prev_convs.items():
                    r_conv_ids = [conv_id for conv_id in convs.keys()][-personalization_r:]
                    prev_convs[user_id] = {
                        conv_id: conv_data for conv_id, conv_data in convs.items() if conv_id in r_conv_ids
                    }

            combined_prev_interactions, used_conv_ids = combine_previous_interactions(
                prev_convs, interaction_type='conv'
            )

        elif 'turns' in personalization_scheme:
            prev_turns_df = prev_convs_df.drop_duplicates(subset=['user_id', 'text'])  # Remove duplicate texts per user
            if 'random' in personalization_scheme:
                # Sample up to 5 texts per user (or take all if less than 5)
                prev_turns_df = prev_turns_df.groupby('user_id', group_keys=False).apply(
                    lambda x: x.sample(n=min(len(x), personalization_r), random_state=42)
                )
            elif 'recency' in personalization_scheme:
                prev_turns_df = prev_turns_df.sort_values(
                    by=['user_id', 'timestamp'], ascending=[True, True]
                )  # Sort by user and latest timestamp
                prev_turns_df = prev_turns_df.groupby('user_id', group_keys=False).tail(
                    personalization_r
                )  # Take top 'r' per user
            # elif "retrieve" in personalization_scheme:
            #     prev_turns_df = prev_turns_df.sort_values(by=['user_id', 'timestamp'], ascending=[True, True])  # Sort by user and latest timestamp
            #     prev_turns_df = prev_turns_df.groupby('user_id', group_keys=False).tail(personalization_pool_size)  # Gather pool of most recent conversations
            #     if "jaccard" in personalization_scheme:
            #         similarity_fn = compute_jaccard_similarity
            #     elif "edit_distance" in personalization_scheme:
            #         similarity_fn = compute_edit_distance_similarity
            # reorder the messages according to the distance metric
            # for row in prev_turns_df.iterrows():
            #     prev_turns_df['dist'] =

            combined_prev_interactions, used_conv_ids = combine_previous_interactions(
                prev_turns_df, interaction_type='turn'
            )

        latest_interactions.loc[:, 'used_prev_conv_ids'] = latest_interactions['user_id'].map(used_conv_ids)
        latest_interactions.loc[:, 'used_prev_conv_count'] = latest_interactions['used_prev_conv_ids'].apply(
            lambda x: len(x) if isinstance(x, list) else 0
        )
        # prepend previous interactions to prefix
        latest_interactions.loc[:, 'prefix'] = latest_interactions.apply(
            lambda row: combined_prev_interactions[row['user_id']] + row['prefix'], axis=1
        )
        # take the first turn of the last conversation
        dataset = latest_interactions[latest_interactions['turn'] == 1].reset_index(drop=True)
        return dataset


if __name__ == '__main__':
    prepare_base_dataset('base_dataset.csv', conv_sim_threshold=0.85)
