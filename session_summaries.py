# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

import json
import string

from boto_session import Args, BotoClient
from botocore.config import Config

PRINTABLE = set(string.printable)


REGION = 'us-east-1'
BEDROCK_CLIENT = BotoClient(Args.make('bedrock-runtime', config=Config(region_name=REGION)))


def make_prompt(dialog, continuation, suggestion):
    with open('./llmau/prompt.txt') as f:
        prompt = f.read()

    words = json.dumps(''.join(c for c in suggestion if c not in '!?.,;[]()"\':').split())

    return (
        prompt
        + f"""

Inputs:
        "dialog": {dialog},
        "original_completion": {continuation},
        "alernative_completion": {suggestion},
        "alernative_completion_words": {words}

Output:"""
    )


def retry(f):
    import time

    def loop(*args, **kw):
        while True:
            try:
                return f(*args, **kw)
            except Exception as ex:
                if 'ThrottlingException' in str(ex):
                    time.sleep(10)
                    continue
                raise

    return loop


def claude_bedrock_args(sys_prompt, prompt, max_tokens, temperature, top_k, top_p):
    if temperature is not None and top_p < 1:
        raise Exception('Cannot sepcify both temperature and top_p')

    if temperature is not None:
        randomness = {'temperature': temperature}
    else:
        randomness = {'top_p': top_p}

    return {
        # "modelId": "anthropic.claude-3-opus-20240229-v1:0",
        'modelId': 'anthropic.claude-3-sonnet-20240229-v1:0',
        # "modelId": "anthropic.claude-3-5-sonnet-20240620-v1:0",
        'contentType': 'application/json',
        'accept': '*/*',
        'body': json.dumps(
            dict(
                messages=[{'role': 'user', 'content': prompt}],
                top_k=top_k,
                max_tokens=max_tokens,
                anthropic_version='bedrock-2023-05-31',
                system=sys_prompt,
                **randomness,
            )
        ),
    }


# @retry
def predict_claude_3(sys_prompt, prompt, max_tokens=100, temperature=0.0, top_k=1, top_p=1):
    args = claude_bedrock_args(sys_prompt, prompt, max_tokens, temperature, top_k, top_p)
    response = BEDROCK_CLIENT.invoke_model(**args)
    response = json.loads(response.get('body').read())
    return response['content'][0]['text'].strip()


def mistral_bedrock_args(prompt, max_tokens, temperature, top_k, top_p):
    return {
        'modelId': 'mistral.mistral-large-2402-v1:0',
        'contentType': 'application/json',
        'accept': 'application/json',
        'body': json.dumps({
            'prompt': prompt,
            'max_tokens': max_tokens,
            'top_k': top_k,
            'top_p': top_p,
            'temperature': temperature,
            'stop': ['\n'],
        }),
    }


def predict_mistral_large(prompt, max_tokens=64, temperature=0, top_k=1, top_p=1):
    args = mistral_bedrock_args(prompt, max_tokens, temperature, top_k, top_p)
    response = BEDROCK_CLIENT.invoke_model(**args)
    response = json.loads(response.get('body').read())
    return response['outputs'][0]['text'].strip()


def brackets(s, opening='{', closing='}'):
    from pyparsing import nestedExpr, originalTextFor

    curly_braces = originalTextFor(nestedExpr(opening, closing))

    parsed = curly_braces.searchString(s)

    return parsed[0][0]


def concat(s1, s2):
    return (s1.strip() + ' ' + s2.strip()).strip()


def printable(s):
    return ''.join([x for x in s if x in PRINTABLE]).strip()


def ask(dialog, continuation, suggestion):
    continuation, suggestion = [s.strip() for s in [continuation, suggestion]]
    dialog, continuation, suggestion = [json.dumps(x) for x in [dialog, continuation, suggestion]]

    prompt = make_prompt(dialog, continuation, suggestion)
    answer = predict_claude_3(prompt)
    print(prompt)
    print(answer)

    try:
        return brackets(answer, '[', ']')
    except Exception:
        print('*' * 10 + 'PROMPT' + '*' * 10)
        print(prompt)
        print('*' * 10 + 'ANSWER' + '*' * 10)
        print(answer)
        raise
