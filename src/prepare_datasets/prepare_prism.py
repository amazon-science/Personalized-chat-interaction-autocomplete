# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

import os
import random
import re

import numpy as np
import pandas as pd
from Levenshtein import distance as levenshtein_distance

from datasets import load_dataset

np.random.seed(42)
random.seed(42)
CHAR_LIMIT = 25000


def build_intro(row):
    """
    Constructs an introduction based on user's demographic information.
    """
    intro = (
        f'I am a {row["age"]} {row["gender"]} from {row["location"]["reside_country"]}.\n'
        + f'My marital status is: {row["marital_status"].lower()}, My education is {row["education"].lower()}, and my current employment status is {row["employment_status"].lower()}.\n'
        + f'My ethnicity is {row["ethnicity"]["simplified"]} and my religion is {row["religion"]["simplified"]}.\n'
        + f'{row["self_description"].strip()}'
    )
    return intro


def get_prompt(conversation_history, end_idx):
    """
    Gets the conversation history up to end_idx as a formatted string.
    Similar to the function in prepare_wildchat.py.
    """
    history = ''
    for i in range(end_idx):
        user_content = conversation_history[i]['user_content']
        model_content = conversation_history[i]['model_content']
        history += f'<|prompter|> {user_content}\n'
        history += f'<|assistant|> {model_content}\n'
    return history


def build_user_prompts(dataset):
    """
    Builds user prompts from the dataset following a similar approach to prepare_wildchat.py.
    Returns a DataFrame with user prompts.
    """
    print('Building user prompts')
    user_prompts = []

    total = len(dataset)
    for idx, (_, row) in enumerate(dataset.iterrows()):
        if idx % 100 == 0:
            print(f'Processing row {idx}/{total}')

        # Extract conversation data
        conversation_id = row['conversation_id']
        user_id = row['user_id']
        timestamp = row['generated_datetime']
        n_turns = row['conversation_turns']
        conversation_history = row['conversation_history']

        # Process conversation history
        structured_history = []

        # Group by turn
        for turn_idx in range(n_turns):
            # Find user message for this turn
            user_messages = [msg for msg in conversation_history if msg['turn'] == turn_idx and msg['role'] == 'user']
            if not user_messages:
                continue
            user_content = user_messages[0]['content']

            # Find model responses for this turn
            model_messages = [msg for msg in conversation_history if msg['turn'] == turn_idx and msg['role'] == 'model']
            if not model_messages:
                continue

            # Select chosen response or highest-scored one
            chosen_responses = [msg for msg in model_messages if msg.get('if_chosen')]
            if chosen_responses:
                model_content = chosen_responses[0]['content']
            else:
                # Find highest-scored response
                max_score = -float('inf')
                max_score_content = ''
                for msg in model_messages:
                    score = msg.get('score')
                    if score is not None and score > max_score:
                        max_score = score
                        max_score_content = msg['content']
                model_content = max_score_content if max_score_content else model_messages[0]['content']

            structured_history.append({'turn': turn_idx, 'user_content': user_content, 'model_content': model_content})

        # Create prompts for each user turn
        for turn_idx, turn_data in enumerate(structured_history):
            history = get_prompt(structured_history[:turn_idx], turn_idx)
            text = turn_data['user_content']
            user_prompts.append({
                'user_id': user_id,
                'cid': conversation_id,
                'mid': f'{conversation_id}/{turn_idx}',
                'timestamp': timestamp,
                'turn': turn_idx + 1,
                'text': text,
                'history': history,
            })

    return pd.DataFrame(user_prompts)


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


def remove_similar_older_rows(df, threshold):
    """
    Removes only the older row in each 'user_id' group when two texts exceed 'threshold' similarity.
    """
    print('Removing similar rows with threshold:', threshold)

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
        indices = group.cid.tolist()

        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):  # Only compare later rows to earlier ones
                if indices[i] == indices[j]:
                    continue
                similarity = compute_edit_distance_similarity(texts[i], texts[j])

                if similarity > threshold:
                    to_remove.add(indices[i])  # Remove earlier row (i)

        return group.loc[~group.cid.isin(to_remove)]  # Keep non-removed conversations

    # Apply filtering to each group
    filtered_df = df.groupby('user_id', group_keys=False).apply(filter_group)

    print(f'Filtered from {len(df)} to {len(filtered_df)} prompts')
    print(
        f'Removed {len(df) - len(filtered_df)} similar prompts ({((len(df) - len(filtered_df)) / len(df) * 100):.2f}%)'
    )

    return filtered_df


def build_word_prefixes(user_prompts):
    """
    Creates word-level prefixes for autocomplete from user prompts.
    Similar to the function in prepare_wildchat.py.
    """
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
    """
    Builds prefixes and filters out empty ones.
    Similar to the function in prepare_wildchat.py.
    """
    user_prompts_df_w_prefix = build_word_prefixes(user_prompts)

    # Remove rows where there is no history and prefix is empty
    user_prompts_df_w_prefix = user_prompts_df_w_prefix[
        user_prompts_df_w_prefix['prefix'] != '<|prompter|> '
    ].reset_index(drop=True)

    return user_prompts_df_w_prefix


def add_user_introductions(df, survey_dataset):
    """
    Adds user introduction to the prefixes.
    """
    print('Adding user introductions to prefixes')

    # Create a function to get intro for each user_id
    def get_intro(user_id):
        matching_rows = survey_dataset[survey_dataset['user_id'] == user_id]
        if not matching_rows.empty:
            return matching_rows.iloc[0]['intro']
        return ''  # Default empty intro if user_id not found

    # Add intro to each user's data
    df['user_intro'] = df['user_id'].apply(get_intro)

    # Prepend intros to prefixes
    df['prefix'] = df['user_intro'] + '\n\n' + df['prefix']

    return df


def split_dataset(user_prompts_df, val_ratio=0.2):
    """
    Splits the dataset into training and validation sets based on user_id.
    """
    print('Splitting dataset into train/val sets')
    unique_users = user_prompts_df['user_id'].unique()
    n_val_users = int(len(unique_users) * val_ratio)

    # Deterministically select validation users
    val_users = sorted(unique_users)[:n_val_users]

    val_set = user_prompts_df[user_prompts_df['user_id'].isin(val_users)].reset_index(drop=True)
    train_set = user_prompts_df[~user_prompts_df['user_id'].isin(val_users)].reset_index(drop=True)

    return train_set, val_set


def filter_users_with_min_conversations(user_prompts_df, min_convs=6):
    """
    Filters out users with fewer than the specified minimum number of conversations.
    """
    user_conv_count = user_prompts_df.groupby('user_id')['cid'].nunique()
    users_with_enough_convs = user_conv_count[user_conv_count >= min_convs].index
    filtered_df = user_prompts_df[user_prompts_df['user_id'].isin(users_with_enough_convs)].reset_index(drop=True)

    print(
        f'Filtered from {user_prompts_df["user_id"].nunique()} to {filtered_df["user_id"].nunique()} users with at least {min_convs} conversations'
    )
    print(f'Filtered from {len(user_prompts_df)} to {len(filtered_df)} prompts')

    return filtered_df


def get_first_turn_of_last_conversation(user_prompts_df):
    """
    For each user, get only the first turn of their last conversation.
    """
    # Sort by timestamp to identify last conversation for each user
    user_prompts_df = user_prompts_df.sort_values(by=['user_id', 'timestamp']).reset_index(drop=True)

    # Get the ID of the last conversation for each user
    latest_conv_ids = user_prompts_df.groupby('user_id')['cid'].last().reset_index()
    latest_conv_map = dict(zip(latest_conv_ids['user_id'], latest_conv_ids['cid']))

    # Filter to keep only the last conversation for each user
    last_convs_df = user_prompts_df[
        user_prompts_df.apply(lambda row: row['cid'] == latest_conv_map[row['user_id']], axis=1)
    ]

    # Further filter to keep only the first turn of each conversation
    last_convs_first_turn = last_convs_df[last_convs_df['turn'] == 1].reset_index(drop=True)

    return last_convs_first_turn


def get_user_prev_convs(prev_convs_df):
    """
    Creates a nested dictionary of previous conversations for each user.
    """
    prev_convs = (
        prev_convs_df.groupby(['user_id', 'cid'])
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
        .apply(lambda x: {row['cid']: row['conv_data'] for _, row in x.iterrows()})
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
    """
    Concatenates previous conversations up to the character limit.
    """
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
    """
    Concatenates previous turns up to the character limit.
    """
    user_prev_turns = [(conv_id, text) for conv_id, text in zip(user_prev_turns_df.cid, user_prev_turns_df.text)]
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
    return combined_turns, used_conv_ids


def combine_previous_interactions(prev_interactions, interaction_type):
    """
    Combines previous interactions based on the interaction type.
    """
    combined_interactions = {}
    used_conv_ids = {}

    if interaction_type == 'conv':
        for user_id, convs in prev_interactions.items():
            # Use the function to get concatenated prompts and the used conv_ids
            user_combined_convs, user_used_conv_ids = concatenate_convs_with_limit(convs)
            combined_interactions[user_id] = user_combined_convs
            used_conv_ids[user_id] = user_used_conv_ids

    elif interaction_type == 'turn':
        for user_id in prev_interactions.user_id.unique():
            user_combined_turns, user_used_conv_ids = concatenate_turns_with_limit(
                prev_interactions[prev_interactions.user_id == user_id]
            )
            combined_interactions[user_id] = user_combined_turns
            used_conv_ids[user_id] = user_used_conv_ids

    return combined_interactions, used_conv_ids


def personalize_dataset(dataset, personalization_scheme, personalization_r):
    """
    Adds personalization to the dataset based on the specified scheme.
    Handles both conversation history and user intro personalization types.
    """
    # Save the full dataset for conversation history lookup
    full_dataset = dataset.copy()

    # Take the first turn of the last conversation for each user
    dataset = dataset.sort_values(by=['user_id', 'timestamp']).reset_index(drop=True)
    latest_conv_ids = dataset.groupby('user_id', as_index=False)['cid'].last()['cid']
    dataset = dataset[dataset['cid'].isin(latest_conv_ids) & (dataset['turn'] == 1)].reset_index(drop=True)

    # If no personalization, return as is
    if personalization_scheme == 'none':
        return dataset

    # Check if we need to add user intros
    use_intro = 'intro' in personalization_scheme

    # Apply conversation history personalization if needed
    if 'convs' in personalization_scheme or 'turns' in personalization_scheme:
        # Setup conversation history personalization using the full dataset
        # Get all conversations except the latest one for each user
        prev_convs_df = full_dataset[~full_dataset['cid'].isin(latest_conv_ids)]

        if len(prev_convs_df) > 0:  # Only proceed if there are previous conversations
            # Save original prefixes before modification
            dataset.loc[:, 'prefix_before'] = dataset['prefix']

            # Get info about previous conversations
            prev_conv_ids = prev_convs_df.groupby('user_id')['cid'].unique().to_dict()
            dataset.loc[:, 'all_prev_conv_ids'] = dataset['user_id'].map(prev_conv_ids)
            dataset.loc[:, 'all_prev_conv_count'] = dataset['all_prev_conv_ids'].apply(
                lambda x: len(x) if isinstance(x, list) else 0
            )

            if 'convs' in personalization_scheme:
                user_prev_convs = get_user_prev_convs(prev_convs_df)

                # All previous user prompts (without assistant replies)
                if 'all_prev_convs' in personalization_scheme:
                    prev_convs = user_prev_convs
                # Randomly sample r previous conversations
                elif 'random' in personalization_scheme:
                    prev_convs = {}
                    for user_id, convs in user_prev_convs.items():
                        conv_ids = [conv_id for conv_id in convs.keys()]
                        r_conv_ids = random.sample(conv_ids, min(personalization_r, len(conv_ids)))
                        prev_convs[user_id] = {
                            conv_id: conv_data for conv_id, conv_data in convs.items() if conv_id in r_conv_ids
                        }
                # Take prompts from r most recent conversations
                elif 'recency' in personalization_scheme:
                    prev_convs = {}
                    for user_id, convs in user_prev_convs.items():
                        r_conv_ids = [conv_id for conv_id in convs.keys()][-personalization_r:]
                        prev_convs[user_id] = {
                            conv_id: conv_data for conv_id, conv_data in convs.items() if conv_id in r_conv_ids
                        }

                # Combine previous conversation text
                combined_prev_interactions, used_conv_ids = combine_previous_interactions(
                    prev_convs, interaction_type='conv'
                )

            elif 'turns' in personalization_scheme:
                prev_turns_df = prev_convs_df.drop_duplicates(
                    subset=['user_id', 'text']
                )  # Remove duplicate texts per user

                if 'random' in personalization_scheme:
                    # Sample up to personalization_r texts per user
                    prev_turns_df = prev_turns_df.groupby('user_id', group_keys=False).apply(
                        lambda x: x.sample(n=min(len(x), personalization_r), random_state=42)
                    )
                elif 'recency' in personalization_scheme:
                    prev_turns_df = prev_turns_df.sort_values(by=['user_id', 'timestamp'], ascending=[True, True])
                    prev_turns_df = prev_turns_df.groupby('user_id', group_keys=False).tail(personalization_r)

                # Combine previous turns text
                combined_prev_interactions, used_conv_ids = combine_previous_interactions(
                    prev_turns_df, interaction_type='turn'
                )

            # Record what we used
            dataset.loc[:, 'used_prev_conv_ids'] = dataset['user_id'].map(used_conv_ids)
            dataset.loc[:, 'used_prev_conv_count'] = dataset['used_prev_conv_ids'].apply(
                lambda x: len(x) if isinstance(x, list) else 0
            )

            # Prepend previous interactions to prefix
            dataset.loc[:, 'prefix'] = dataset.apply(
                lambda row: combined_prev_interactions[row['user_id']] + row['prefix']
                if row['user_id'] in combined_prev_interactions
                else row['prefix'],
                axis=1,
            )

    # Apply user intro personalization if needed
    if use_intro:
        survey_dataset = load_dataset('HannahRoseKirk/prism-alignment', 'survey', revision='main')['train'].to_pandas()
        print('Creating user introductions')
        survey_dataset['intro'] = survey_dataset.apply(build_intro, axis=1)
        dataset = add_user_introductions(dataset, survey_dataset)

    return dataset


def prepare_base_dataset(save_path, conv_sim_threshold=0.85):
    """
    Prepares the base PRISM dataset, splits it into train/val sets,
    and saves them to files.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Load the PRISM dataset
    print('Loading PRISM dataset')
    conv_dataset = load_dataset('HannahRoseKirk/prism-alignment', 'conversations', revision='main')['train'].to_pandas()

    # Build user prompts
    print('Building user prompts')
    user_prompts_df = build_user_prompts(conv_dataset)

    # Filter users with fewer than 6 conversations
    print('Filtering users with fewer than 6 conversations')
    filtered_prompts_df = filter_users_with_min_conversations(user_prompts_df, min_convs=6)

    # Remove similar rows (rephrased messages) using edit distance similarity
    print('Removing similar rows based on edit distance')
    print(f'Num prompts before removing similar rows: {len(filtered_prompts_df)}')
    filtered_prompts_df = remove_similar_older_rows(filtered_prompts_df, conv_sim_threshold)
    print(f'Num prompts after removing similar rows: {len(filtered_prompts_df)}')

    # Re-filter users with fewer than 6 conversations after similarity filtering
    print('Re-filtering users with fewer than 6 conversations after similarity filtering')
    filtered_prompts_df = filter_users_with_min_conversations(filtered_prompts_df, min_convs=6)

    # Build prefixes from the filtered prompts
    print('Building prefixes')
    prefixes_df = build_prefixes(filtered_prompts_df)

    # Split into train/val sets
    print('Splitting into train/val sets')
    train_df, val_df = split_dataset(prefixes_df)

    # Save base dataset files
    train_df.to_csv(os.path.join(save_path, 'base_train.csv'), index=False)
    val_df.to_csv(os.path.join(save_path, 'base_val.csv'), index=False)

    print(f'Base dataset preparation complete. Files saved to {save_path}')

    # Print dataset statistics
    print('\nBase Dataset Statistics:')
    print(f'Total users: {prefixes_df["user_id"].nunique()}')
    print(f'Total conversations: {prefixes_df["cid"].nunique()}')
    print(f'Total prefixes: {len(prefixes_df)}')
    print(f'Train prefixes: {len(train_df)}')
    print(f'Val prefixes: {len(val_df)}')

    return train_df, val_df


if __name__ == '__main__':
    # Default settings
    save_dir = 'prism_dataset'
    conv_sim_threshold = 0.85
    personalization_scheme = 'recency_turns'
    personalization_r = 1000

    # Step 1: Prepare base dataset
    train_df, val_df = prepare_base_dataset(save_path=save_dir, conv_sim_threshold=conv_sim_threshold)

    # Step 2: Apply personalization
    if personalization_scheme != 'none':
        print(f'\nApplying personalization scheme: {personalization_scheme}')
        val_personalized = personalize_dataset(
            val_df,
            personalization_scheme=personalization_scheme,
            personalization_r=personalization_r,
        )

        # Save personalized datasets
        # train_personalized.to_csv(os.path.join(save_dir, "personalized_train.csv"), index=False)
        val_personalized.to_csv(os.path.join(save_dir, 'personalized_val.csv'), index=False)

        # Print personalization statistics
        print('\nPersonalized Dataset Statistics:')
        # print(f"Total users: {len(set(train_personalized['user_id']).union(set(val_personalized['user_id'])))} ")
        # print(f"Train users: {train_personalized['user_id'].nunique()}")
        print(f'Val users: {val_personalized["user_id"].nunique()}')
        # print(f"Total prefixes: {len(train_personalized) + len(val_personalized)}")
        # print(f"Train prefixes: {len(train_personalized)}")
        print(f'Val prefixes: {len(val_personalized)}')
