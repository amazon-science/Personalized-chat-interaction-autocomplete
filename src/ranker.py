# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

import pickle
import random
import re
from os import listdir
from os.path import isfile, join

import numpy as np
import torch
from Levenshtein import distance as levenshtein_distance
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from llm_ranker import LlmRankerForPrism

LAYER_INDICES = [31, 27, 23, 19, 15, 11]


def parse_prefix(prefix):
    """
    Extracts previous messages and the current message from the formatted input.

    Args:
        prefix (str): The formatted string containing multiple messages.

    Returns:
        tuple: (curr_prefix, prev_turns) where
               - curr_prefix is the text after "## Current Message ##"
               - prev_turns is a list of previous messages
    """
    # Split messages using regex to detect "## Message X ##" headers
    messages = re.split(r'## .*? ##', prefix)

    # Remove empty splits due to leading/trailing spaces
    messages = [msg.strip() for msg in messages if msg.strip()]

    assert len(messages) > 1

    # Previous messages (everything except the last one)
    prev_turns = [turn.strip() for msg in messages[:-1] for turn in msg.split('<|prompter|>') if turn.strip()]

    # The current message (last extracted section)
    curr_prefix = messages[-1]

    return curr_prefix, prev_turns


def get_best_gpu():
    """
    Returns the best available GPU (least memory used), or "cpu" if no GPU is available.
    """
    if not torch.cuda.is_available():
        return 'cpu'  # No GPU found

    best_gpu = None
    min_memory_used = float('inf')

    for i in range(torch.cuda.device_count()):
        memory_used = torch.cuda.memory_reserved(i)  # Memory in bytes
        if memory_used < min_memory_used:
            min_memory_used = memory_used
            best_gpu = i  # Choose the GPU with the least reserved memory

    return f'cuda:{best_gpu}' if best_gpu is not None else 'cpu'


def compute_jaccard_similarity(text1, text2):
    """
    Computes the similarity based on Jaccard distance.
    Returns a similarity score between 0 and 1 (1 means identical).
    """
    set_1, set_2 = (
        set(text1.replace('<|prompter|> ', '').split()),
        set(text2.replace('<|prompter|> ', '').split()),
    )  # Tokenize into words
    intersection = len(set_1 & set_2)
    union = len(set_1 | set_2)
    return intersection / union if union != 0 else 0  # Avoid division by zero


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


class Ranker:
    def __init__(
        self,
        saved_completions,
        confidence_measure='',
        personal_w=0.0,
        model_name=None,
        mlp_path=None,
        dataset=None,
        root_path=None,
    ):
        self.saved_completions = saved_completions
        self.confidence_measure = confidence_measure
        self.rank_fn = getattr(self, self.confidence_measure)
        self.root_path = root_path or './'
        random.seed(42)
        if 'personal_mistral_rank' in confidence_measure:
            self.mistral_ranker = LlmRankerForPrism()
            self.personal = True
        elif 'personal' in confidence_measure:
            self.personal = True
            self.personal_w = float(personal_w)
            completions_dir_path = f'{self.root_path}/completions/{dataset}/word/{model_name.replace("/", "_")}'
            with open(
                join(
                    completions_dir_path,
                    [
                        f
                        for f in listdir(completions_dir_path)
                        if isfile(join(completions_dir_path, f)) and ('all_prev_convs' in f)
                    ][0],
                ),
                'rb',
            ) as cf:
                all_prev_convs_completions = pickle.load(cf)
            # store embeddings of previous messages
            if 'cosine' in confidence_measure:
                # Get the best GPU
                device = get_best_gpu()
                print(f'Using device: {device}')
                self.embed_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
                self.user_embeddings = {}
                print('Initializing user embeddings')
                for i, c in enumerate(tqdm(all_prev_convs_completions)):
                    user_id = c['dataset_row']['user_id']
                    if user_id not in self.user_embeddings.keys():
                        _, prev_turns = parse_prefix(c['dataset_row']['prefix'])
                        prev_turns_embeddings = self.embed_model.encode(prev_turns, convert_to_numpy=True)
                        self.user_embeddings[user_id] = prev_turns_embeddings
            # store previous messages as is
            else:
                self.user_prev_messages = {}
                for i, c in enumerate(tqdm(all_prev_convs_completions)):
                    user_id = c['dataset_row']['user_id']
                    if user_id not in self.user_prev_messages.keys():
                        _, prev_turns = parse_prefix(c['dataset_row']['prefix'])
                        self.user_prev_messages[user_id] = prev_turns
        else:
            self.personal = False

    def rank(self, prefix, completions, user_id=None):
        if self.personal:
            completions = self.rank_fn(prefix, completions, user_id)
        else:
            completions = self.rank_fn(prefix, completions)
        return self.sort_by_key(completions, self.confidence_measure)

    def sort_by_key(self, completions, key):
        sorted_completions = dict(
            sorted(
                completions.items(),
                key=lambda item: item[1][key],
                reverse=True,
            )
        )
        return list(sorted_completions.keys()), [c[key] for c in sorted_completions.values()]

    def get_top_ll_completions(self, prefix, completions, n):
        ll_completions = self.log_likelihood(prefix, completions)
        self.ll_sorted_completions, ll_scores = self.sort_by_key(ll_completions, 'log_likelihood')
        # Return top n completions in descending log-likelihood order
        top_n_keys = self.ll_sorted_completions[:n]
        # Use ordered construction to preserve log-likelihood ranking
        ordered_completions = {}
        for key in top_n_keys:
            ordered_completions[key] = completions[key]
        return ordered_completions

    def personal_cosine_mean(self, prefix, completions, user_id):
        completions = self.get_top_ll_completions(prefix, completions, n=5)
        curr_prefix = prefix.split('## Current Conversation ##')[-1].strip()
        for completion, tokens_dict in completions.items():
            suggested_text = curr_prefix + ' ' + completion.strip()
            suggested_text_embedding = self.embed_model.encode(suggested_text, convert_to_numpy=True)
            tokens_dict['personal_cosine_mean'] = cosine_similarity(
                [suggested_text_embedding], [np.mean(self.user_embeddings[user_id], axis=0)]
            )[0][0]
        return completions

    def personal_cosine_max(self, prefix, completions, user_id):
        completions = self.get_top_ll_completions(prefix, completions, n=5)
        curr_prefix = prefix.split('## Current Conversation ##')[-1].strip()
        for completion, tokens_dict in completions.items():
            suggested_text = curr_prefix + ' ' + completion.strip()
            suggested_text_embedding = self.embed_model.encode(suggested_text, convert_to_numpy=True)
            tokens_dict['personal_cosine_max'] = cosine_similarity(
                self.user_embeddings[user_id], suggested_text_embedding.reshape(1, -1)
            ).max()
        return completions

    def personal_jaccard_mean(self, prefix, completions, user_id):
        completions = self.get_top_ll_completions(prefix, completions, n=5)
        curr_prefix = prefix.split('## Current Conversation ##')[-1].strip()
        for completion, tokens_dict in completions.items():
            suggested_text = curr_prefix + ' ' + completion.strip()
            jaccard_similarities = [
                compute_jaccard_similarity(suggested_text, prev_message)
                for prev_message in self.user_prev_messages[user_id]
            ]
            tokens_dict['personal_jaccard_mean'] = np.mean(jaccard_similarities)
        return completions

    def personal_jaccard_max(self, prefix, completions, user_id):
        completions = self.get_top_ll_completions(prefix, completions, n=5)
        curr_prefix = prefix.split('## Current Conversation ##')[-1].strip().replace('<|prompter|> ', '')
        for completion, tokens_dict in completions.items():
            suggested_text = curr_prefix + ' ' + completion.strip()
            jaccard_similarities = [
                compute_jaccard_similarity(suggested_text, prev_message)
                for prev_message in self.user_prev_messages[user_id]
            ]
            tokens_dict['personal_jaccard_max'] = (
                np.max(jaccard_similarities) + self.personal_w * tokens_dict['log_likelihood']
            )
        return completions

    def personal_edit_dist_mean(self, prefix, completions, user_id):
        completions = self.get_top_ll_completions(prefix, completions, n=5)
        curr_prefix = prefix.split('## Current Conversation ##')[-1].strip()
        for completion, tokens_dict in completions.items():
            suggested_text = curr_prefix + ' ' + completion.strip()
            jaccard_similarities = [
                compute_edit_distance_similarity(suggested_text, prev_message)
                for prev_message in self.user_prev_messages[user_id]
            ]
            tokens_dict['personal_edit_dist_mean'] = np.mean(jaccard_similarities)
        return completions

    def personal_edit_dist_max(self, prefix, completions, user_id):
        completions = self.get_top_ll_completions(prefix, completions, n=5)
        curr_prefix = prefix.split('## Current Conversation ##')[-1].strip()
        for completion, tokens_dict in completions.items():
            suggested_text = curr_prefix + ' ' + completion.strip()
            jaccard_similarities = [
                compute_edit_distance_similarity(suggested_text, prev_message)
                for prev_message in self.user_prev_messages[user_id]
            ]
            tokens_dict['personal_edit_dist_max'] = np.max(jaccard_similarities)
        return completions

    def neg_entropy(self, tokens_logprobs, normalize):
        max_len = max(len(arr) for arr in tokens_logprobs)
        probs = np.exp(np.stack([np.pad(arr, (0, max_len - len(arr)), constant_values=0) for arr in tokens_logprobs]))
        if normalize:
            probs = probs / np.sum(probs, axis=1, keepdims=True)
        neg_entropy = np.sum(probs * np.log(probs), axis=1)
        return neg_entropy

    def neg_perplexity(self, prefix, completions):
        for _, tokens_dict in completions.items():
            tokens_dict['neg_perplexity'] = np.exp(np.mean(tokens_dict['logprobs']))
        return completions

    def log_likelihood(self, prefix, completions):
        for _, tokens_dict in completions.items():
            tokens_dict['log_likelihood'] = np.sum(tokens_dict['logprobs'])
        return completions

    def max_logprob(self, prefix, completions):
        for _, tokens_dict in completions.items():
            tokens_dict['max_logprob'] = np.max(tokens_dict['logprobs'])
        return completions

    def min_logprob(self, prefix, completions):
        for _, tokens_dict in completions.items():
            tokens_dict['min_logprob'] = np.min(tokens_dict['logprobs'])
        return completions

    def mean_neg_entropy_norm(self, prefix, completions):
        for _, tokens_dict in completions.items():
            tokens_dict['mean_neg_entropy_norm'] = np.mean(
                self.neg_entropy(tokens_dict['top_tokens_logprobs'], normalize=True)
            )
        return completions

    def mean_neg_entropy(self, prefix, completions):
        for _, tokens_dict in completions.items():
            tokens_dict['mean_neg_entropy'] = np.mean(
                self.neg_entropy(tokens_dict['top_tokens_logprobs'], normalize=False)
            )
        return completions

    def max_neg_entropy_norm(self, prefix, completions):
        for _, tokens_dict in completions.items():
            tokens_dict['max_neg_entropy_norm'] = np.max(
                self.neg_entropy(tokens_dict['top_tokens_logprobs'], normalize=True)
            )
        return completions

    def min_neg_entropy_norm(self, prefix, completions):
        for _, tokens_dict in completions.items():
            tokens_dict['min_neg_entropy_norm'] = np.min(
                self.neg_entropy(tokens_dict['top_tokens_logprobs'], normalize=True)
            )
        return completions

    def max_neg_entropy(self, prefix, completions):
        for _, tokens_dict in completions.items():
            tokens_dict['max_neg_entropy'] = np.max(
                self.neg_entropy(tokens_dict['top_tokens_logprobs'], normalize=False)
            )
        return completions

    def min_neg_entropy(self, prefix, completions):
        for _, tokens_dict in completions.items():
            tokens_dict['min_neg_entropy'] = np.min(
                self.neg_entropy(tokens_dict['top_tokens_logprobs'], normalize=False)
            )
        return completions

    # TODO: deal with single token output
    def mean_t1_t2_ratio(self, prefix, completions):
        for _, tokens_dict in completions.items():
            probs = np.exp(np.stack(tokens_dict['top_tokens_logprobs']))
            t1 = probs[:, 0]
            t2 = probs[:, 1]
            tokens_dict['mean_t1_t2_ratio'] = (t1 / t2).mean()
        return completions

    def mean_t1_t2_gap(self, prefix, completions):
        for _, tokens_dict in completions.items():
            probs = np.exp(np.stack(tokens_dict['top_tokens_logprobs']))
            t1 = probs[:, 0]
            t2 = probs[:, 1]
            tokens_dict['mean_t1_t2_gap'] = (t1 - t2).mean()
        return completions

    def sequence_length(self, prefix, completions):
        for c, tokens_dict in completions.items():
            tokens_dict['sequence_length'] = -len(c)
        return completions

    def random(self, prefix, completions):
        for _, tokens_dict in completions.items():
            tokens_dict['random'] = random.random()
        return completions

    def personal_mistral_rank(self, prefix, completions, user_id):
        self.mistral_ranker.rank(prefix, completions, user_id, 'personal_mistral_rank')
        return completions

    def personal_dspy_rank(self, prefix, completions, user_id):
        """Use the DSPy MIPRO prompt for ranking completions with user profiles"""
        try:
            # Get top 5 log-likelihood completions first (consistent with other personal rankers)
            completions = self.get_top_ll_completions(prefix, completions, n=5)

            # Load the DSPy ranker if not already loaded
            if not hasattr(self, 'dspy_ranker'):
                self._init_dspy_ranker()

            # Use the DSPy ranker to score completions
            self.dspy_ranker.rank(prefix, completions, user_id, 'personal_dspy_rank')
            return completions
        except Exception as e:
            print(f'Error in DSPy ranking: {e}')
            # Re-raise the exception to avoid silent failures
            raise RuntimeError(f'DSPy ranking failed: {e}')

    def _init_dspy_ranker(self):
        """Initialize the DSPy ranker with MIPRO configuration"""
        try:
            from dspy_ranker import DSPyRankerForPrism

            root_path = getattr(self, 'root_path', './')
            self.dspy_ranker = DSPyRankerForPrism(root_path=root_path)
        except ImportError as e:
            raise ImportError(
                f'Failed to import DSPy ranker: {e}. Make sure dspy_ranker.py is available and DSPy dependencies are installed.'
            )

    def oracle_ranker(self, completions, prefix, ground_truth, metrics):
        accepted_text = metrics.exact_match_acceptance(prefix, completions, prefix.strip() + ' ' + ground_truth)
        if accepted_text is None:
            return [], []
        return [accepted_text], [1.0]
