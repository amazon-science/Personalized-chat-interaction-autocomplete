# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

import os
import pickle
import re
import subprocess
import time

import docker
import pandas as pd
import requests
import tqdm


def count_words(text):
    if pd.isna(text):
        return 0
    return len(text.split())


def truncate_history(text, trunc):
    if len(text) < trunc:
        return text

    trunc += text[-trunc:].count(('<|prompter|> ')) * len('<|prompter|> ')

    # take end of text
    res = text[-trunc:]
    # get start of next word
    first_word = len(re.search(r'^.*?[\s]+|.*$', res, flags=re.DOTALL).group())
    return res[first_word:]


class TGIRunner:
    def __init__(self, model, inference_params, tgi_image, work_path, hf_token=None, gpu_id=0):
        self.port = 8080 + gpu_id
        self.inference_params = (
            inference_params
            if 'mamba' not in model
            else {k: v for k, v in inference_params.items() if k != 'top_n_tokens'}
        )
        self.max_input_tokens = 29000 if ('gemma' not in model and 'gpt' not in model and '70B' not in model) else 2048
        self.headers = {
            'Content-Type': 'application/json',
        }
        self.tgi_endpoint = f'http://127.0.0.1:{self.port}/generate'
        self.hf_token = hf_token or ''
        self.tgi_image = tgi_image
        self.work_path = work_path
        self.gpu_id = gpu_id
        self.model = model
        self.stop_symbols = ['.', '!', '?', '\n', '<|']
        if self.model == 'meta-llama/Llama-3.1-70B-Instruct':
            self.stop_symbols += [
                '<|start_header_id|>',
                '<|end_header_id|>',
                '<|eot_id|>',
                '<|reserved_special_token',
            ]
        self.start_container()

    def start_container(self):
        # force kill existing containers running on the same port
        existing_containers = subprocess.run(
            ['docker', 'ps', '-q'],
            capture_output=True,
            text=True,
        ).stdout.strip()
        for container in existing_containers.split('\n'):
            port = subprocess.run(['docker', 'port', container], capture_output=True, text=True).stdout.strip()
            if str(self.port) in port:
                print(f'found existing container running on port {self.port}, stopping it.')
                subprocess.run(
                    ['docker', 'stop', container],
                    capture_output=True,
                    text=True,
                )

        environment_vars = [
            f'HF_TOKEN={self.hf_token}',
            'MAX_BEST_OF=10',
            'MAX_CONCURRENT_REQUESTS=10',
            'MAX_TOP_N_TOKENS=10',
            'MAX_STOP_SEQUENCES=10',
            'TRUST_REMOTE_CODE=true',
            'MAX_BATCH_SIZE=1',
        ]

        device_ids = (
            [str(self.gpu_id)] if self.model != 'meta-llama/Llama-3.1-70B-Instruct' else [str(i) for i in range(8)]
        )

        client = docker.from_env()
        self.container = client.containers.run(
            self.tgi_image,
            command=f'--model-id {self.model}',
            volumes={self.work_path: {'bind': '/data', 'mode': 'rw'}},
            ports={'80/tcp': self.port},
            shm_size='1g',
            device_requests=[docker.types.DeviceRequest(device_ids=device_ids, capabilities=[['gpu']])],
            environment=environment_vars,
            detach=True,
            stdout=True,
        )

        time.sleep(10)
        container_logs = self.container.logs().decode('utf-8')
        i = 1
        max_attempts = 60
        retry_interval = 20
        while container_logs.find('defaulting to 0.0.0.0') == -1 and container_logs.find('Connected') == -1:
            print(f'container start attempt {i}')
            if i > max_attempts:
                print(f'failed to start container after {max_attempts * retry_interval} secs')
                print(container_logs)
                exit(1)
            if container_logs.find('download') == -1:
                print(container_logs)
                raise AssertionError('TGI endpoint failed to start')
            time.sleep(retry_interval)
            i += 1
            container_logs = self.container.logs().decode('utf-8')

        if self.model in ['meta-llama/Llama-3.1-70B-Instruct', 'microsoft/Phi-3.5-mini-instruct']:
            time.sleep(120)

    def run_inference(self, prompt):
        body = {
            'inputs': prompt,
            'parameters': {
                'do_sample': True if self.inference_params['best_of'] > 1 else False,
                'stop': self.stop_symbols,
                'details': True,
                **self.inference_params,
                'top_p': None if self.inference_params['top_p'] == 1 else self.inference_params['top_p'],
            },
        }
        try:
            response = requests.post(self.tgi_endpoint, headers=self.headers, json=body, timeout=120)
            return response
        except requests.exceptions.Timeout:
            print('The request timed out after 2 minutes.')
            response = None
        return response

    def run_dataset(self, dataset, save_path='', local_model_path='', instruct=False, existing_completions=[]):
        os.makedirs(''.join(save_path[: save_path.rindex('/')]), exist_ok=True)
        completions = existing_completions
        start_idx = len(completions)

        for i in tqdm.tqdm(range(start_idx, len(dataset))):
            row = dataset.iloc[i]
            prompt = row['prefix']
            if self.inference_params['history_len'] is not None:
                prompt = truncate_history(prompt, self.inference_params['history_len'])
            # TODO: skip these for now, but need to understand why it fails
            if self.model == '/data/model_files' and row.get('mid', None) == '42f079f2-f3ca-40ed-b168-cfe7092c447c':
                completions.append({'dataset_row': row.to_dict(), 'completion': {}})
                continue
            try:
                response = self.run_inference(prompt)
                completion = response.json()
                if 'error' in completion.keys():
                    raise Exception(completion['error'])
                completions.append({
                    'dataset_row': row.to_dict(),
                    'prompt': prompt,
                    'completion': completion,
                    'inference_time': response.headers['x-inference-time'],
                })
            except Exception as e:
                print(f'Failed to get completion for prefix {i}, {e}')
                if 'Input validation error' in str(e):
                    completions.append({'dataset_row': row.to_dict(), 'completion': {}})
                else:
                    print(self.container.logs().decode('utf-8')[-6000:])
                    exit(1)
            if i % 100 == 0 and i > 0:
                f = open(save_path, 'wb')
                pickle.dump(completions, f)
                f.close()

        # save all completions to pkl file
        f = open(save_path, 'wb')
        pickle.dump(completions, f)
        f.close()

        assert len(completions) == len(dataset['prefix']), (
            f'Length mismatch. Prefixes: {len(dataset["prefix"])}, Completions: {len(completions)}.'
        )
