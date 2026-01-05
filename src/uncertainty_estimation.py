# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

import ast
import math
import os

import pandas as pd
import torch
from evaluate import load, logging
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer


class Perplexity:
    def __init__(self, model_id, device=None):
        if device is not None:
            assert device in ['gpu', 'cpu', 'cuda'], 'device should be either gpu or cpu.'
            if device == 'gpu':
                self.device = 'cuda'
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, revision='main').to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision='main')

    def logprob(self, predictions, batch_size: int = 4, add_start_token: bool = True, max_length=None):
        # if batch_size > 1 (which generally leads to padding being required), and
        # if there is not an already assigned pad_token, assign an existing
        # special token to also be the padding token
        if self.tokenizer.pad_token is None and batch_size > 1:
            existing_special_tokens = list(self.tokenizer.special_tokens_map_extended.values())
            # check that the model already has at least one special token defined
            assert len(existing_special_tokens) > 0, (
                'If batch_size > 1, model must have at least one special token to use for padding. Please use a different model or set batch_size=1.'
            )
            # assign one of the special tokens to also be the pad token
            self.tokenizer.add_special_tokens({'pad_token': existing_special_tokens[0]})

        if add_start_token and max_length:
            # leave room for <BOS> token to be added:
            assert self.tokenizer.bos_token is not None, (
                'Input model must already have a BOS token if using add_start_token=True. Please use a different model, or set add_start_token=False'
            )
            max_tokenized_len = max_length - 1
        else:
            max_tokenized_len = max_length

        encodings = self.tokenizer(
            predictions,
            add_special_tokens=False,
            padding=True,
            truncation=True if max_tokenized_len else False,
            max_length=max_tokenized_len,
            return_tensors='pt',
            return_attention_mask=True,
        ).to(self.device)

        encoded_texts = encodings['input_ids']
        attn_masks = encodings['attention_mask']

        # check that each input is long enough:
        if add_start_token:
            assert torch.all(torch.ge(attn_masks.sum(1), 1)), 'Each input text must be at least one token long.'
        else:
            assert torch.all(torch.ge(attn_masks.sum(1), 2)), (
                'When add_start_token=False, each input text must be at least two tokens long. Run with add_start_token=True if inputting strings of only one token, and remove all empty input strings.'
            )

        logprobs = []
        loss_fct = CrossEntropyLoss(reduction='none')

        for start_index in logging.tqdm(range(0, len(encoded_texts), batch_size)):
            end_index = min(start_index + batch_size, len(encoded_texts))
            encoded_batch = encoded_texts[start_index:end_index]
            attn_mask = attn_masks[start_index:end_index]

            if add_start_token:
                bos_tokens_tensor = torch.tensor([[self.tokenizer.bos_token_id]] * encoded_batch.size(dim=0)).to(
                    self.device
                )
                encoded_batch = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
                attn_mask = torch.cat(
                    [torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(self.device), attn_mask], dim=1
                )

            labels = encoded_batch

            with torch.no_grad():
                out_logits = self.model(encoded_batch, attention_mask=attn_mask).logits

            shift_logits = out_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

            logprob = loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch
            logprobs += logprob.tolist()

        return {'logprobs': logprobs}

    def perplexity_w_prefix(self, prefix, text, model_id):
        assert text.startswith(prefix), 'text must start with prefix'
        perplexity = load('perplexity', module_type='metric')
        prefix_perp, text_perp = perplexity.compute(predictions=[prefix, text], model_id=model_id)['perplexities']
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision='main')
        prefix_tokens, text_tokens = [len(ids) for ids in tokenizer([prefix, text])['input_ids']]
        delta_tokens = text_tokens - prefix_tokens
        print(prefix_tokens, text_tokens, delta_tokens)
        return ((text_perp**text_tokens) / (prefix_perp**prefix_tokens)) ** (1 / delta_tokens)

    def perplexity_w_prefix_efficient(self, prefix, completion, k=None):
        if k is not None:
            completion = ' '.join(completion.split()[:k])
        text = prefix + ' ' + completion
        text_prob = self.logprob(predictions=[text])['logprobs'][0]
        prefix_tokens, text_tokens = [len(ids) for ids in self.tokenizer([prefix, text])['input_ids']]
        delta_tokens = text_tokens - prefix_tokens
        completion_text = text_prob[-delta_tokens:]
        return math.exp(sum(completion_text) / len(completion_text))  # mean nll


class UncertaintyEstimation:
    def __init__(self, model_id, device=None):
        if device is not None:
            assert device in ['gpu', 'cpu', 'cuda'], 'device should be either gpu or cpu.'
            if device == 'gpu':
                self.device = 'cuda'
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision='main')

    def get_logprobs_per_word(self, logprobs):
        logprobs_per_word = []
        curr_logprobs = (logprobs[0][0], [logprobs[0][1]])
        for tok, logprob in logprobs[1:]:
            # new word
            if tok.startswith((' ', ',', '.', '?', '!')):
                logprobs_per_word.append(curr_logprobs)
                curr_logprobs = (tok, [logprob])
            # append to current word
            else:
                curr_logprobs = (curr_logprobs[0] + tok, curr_logprobs[1] + [logprob])
        logprobs_per_word.append(curr_logprobs)
        return logprobs_per_word

    def confidence_scores(self, row, max_k=20, save_path=None):
        pred = row['gen_completion']
        if pred == '':
            return

        if max_k is None:
            max_k = len(row['gen_completion'].split())

        # get logprobs
        logprobs_data = ast.literal_eval(row['gen_logprobs'])
        logprobs = [(t['text'], t['logprob']) for t in logprobs_data]
        logprobs_per_word = self.get_logprobs_per_word(logprobs)

        # perplexity
        perplexity = {}
        for k in range(1, max_k + 1):
            logprob_sum = [sum(logprob[1]) for logprob in logprobs_per_word[:k]]
            perplexity[f'perplexity-{k}'] = math.exp(-1 * sum(logprob_sum) / len(logprob_sum))

        # min logprob
        min_logprob = {}
        for k in range(1, max_k + 1):
            min_logprob[f'min_logprob-{k}'] = min([min(logprob[1]) for logprob in logprobs_per_word[:k]])

        for k in range(1, max_k + 1):
            min_logprob[f'min_logprob_norm-{k}'] = min([min(logprob[1]) for logprob in logprobs_per_word[:k]]) / len(
                logprobs_per_word[:k]
            )

        return pd.Series({**perplexity, **min_logprob})

    def run_confidence_scores(self, model_id, results, save_path):
        results_metrics = pd.merge(
            results,
            results[['prefix', 'gt_completion', 'gen_completion', 'gen_logprobs']].progress_apply(
                self.confidence_scores, axis=1
            ),
            left_index=True,
            right_index=True,
        )

        os.makedirs(save_path, exist_ok=True)
        results_metrics.to_csv(
            os.path.join(save_path, f'metrics_{model_id.replace("/", "_")}.csv'),
            index=False,
        )

        return results_metrics
