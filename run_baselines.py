# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

import argparse
import os
import pickle
import time

import docker
import pandas as pd

from src.inference import TGIRunner, subprocess
from src.prepare_datasets.prepare_wildchat import personalize_dataset, prepare_base_dataset

TGI_IMAGE = 'ghcr.io/huggingface/text-generation-inference:3.1.0'
VOLUME = os.path.join(os.getcwd(), 'runtime', '{model}')
LOCAL_ROOT_PATH = './'
S3_ROOT_PATH = '<s3_root_path>'
hf_token_path = os.path.join(LOCAL_ROOT_PATH, 'resources/hf_token.txt')


config_parser = parser = argparse.ArgumentParser(description='inference args', add_help=False)
parser.add_argument(
    '--model_id',
    default='microsoft/Phi-3.5-mini-instruct',
    help='model name as appears in huggingface or path to finetuned model',
)
parser.add_argument('--gpu_id', default=0, type=int, help='gpu id')
parser.add_argument('--best_of', default=5, type=int, help='number of generated output sequences')
parser.add_argument('--max_new_tokens', default=20, type=int, help='save logprob data for top n tokens')
parser.add_argument('--top_n_tokens', default=0, type=int, help='save logprob data for top n tokens')
parser.add_argument('--temperature', default=1.0, type=float, help='')
parser.add_argument('--top_p', default=1.0, type=float, help='')
parser.add_argument('--prefixes', default='word', type=str, help='complete every char or word')
parser.add_argument('--dataset', default='wildchat', type=str, help='dataset to run inference on')
parser.add_argument(
    '--hp_subset', default=None, type=float, help='fraction of dataset to sample for hparam exploration'
)
parser.add_argument('--history_len', default=None, type=int, help='the amount of characters to include in the prompt')
parser.add_argument(
    '--personalization_scheme',
    default='recency_turns',
    type=str,
    help="""(1) none - no personalization
                            (2) all_prev_convs - prepend all previous user conversations (without assistant replies)
                            (3) random_convs - randomly sample r previous user conversations (all turns, no assistant replies)
                            (4) random_turns - randomly sample r previous user turns (no assistant replies)
                            (5) recency_convs - r most recent previous user conversations (all turns, no assistant replies)
                            (6) recency_turns - r most recent previous user turns (no assistant replies)
                            """,
    choices=[
        'none',
        'all_prev_convs',
        'all_prev_convs',
        'random_convs',
        'random_turns',
        'recency_convs',
        'recency_turns',
    ],
)
parser.add_argument(
    '--personalization_r',
    default='20',
    type=int,
    help='How many previous interactions to use for personalization. Relevant only for personalization schemes that require r.',
)
parser.add_argument(
    '--conv_sim_threshold', default=0.85, type=float, help='Similarity threshold for removing rephrased messages.'
)
parser.add_argument(
    '--num_samples',
    default=None,
    type=int,
    help='Number of samples from the dataset to run on. If None, runs on all samples.',
)


def prepare_model(s3_model_path, model_name):
    local_model_path = os.path.join(LOCAL_ROOT_PATH, f'tmp/runtime/{model_name}/model_files/')

    os.makedirs(local_model_path, mode=0o750, exist_ok=True)

    subprocess.run(
        ['aws', 's3', 'sync', s3_model_path, local_model_path],
        capture_output=True,
        text=True,
    )

    return local_model_path


if __name__ == '__main__':
    args = vars(parser.parse_args())
    print(args)

    # prepare dataset
    dataset_path = os.path.join(LOCAL_ROOT_PATH, f'datasets/{args["dataset"]}/')
    dataset_save_name = f'{args["dataset"]}/{args["prefixes"]}'
    if args['hp_subset'] is not None:
        dataset_save_name += f'/subset_{args["hp_subset"]}'
        dataset_path += f'subset_{args["hp_subset"]}'

    if args['dataset'] == 'wildchat':
        dataset_path += f'user_prompts_w_prefix_t={args["conv_sim_threshold"]}.csv'
        dataset_prepare_fn = prepare_base_dataset

    if not os.path.isfile(dataset_path):
        dataset_prepare_fn(
            save_path=dataset_path, conv_sim_threshold=args['conv_sim_threshold'], num_samples=args['num_samples']
        )
    base_dataset = pd.read_csv(dataset_path, index_col=0, nrows=args['num_samples']).reset_index()

    dataset = personalize_dataset(base_dataset, args['personalization_scheme'], args['personalization_r'])

    model_name = args['model_id']
    if 'finetuned' in args['model_id']:
        s3_model_path = '<s3_model_path>'
        model_local_path = prepare_model(s3_model_path, model_name)
        model = '/data/model_files'
    else:
        model = model_name
        model_local_path = None
    model_save_name = model_name.replace('/', '_')

    completions_save_path = (
        os.path.join(LOCAL_ROOT_PATH, f'completions/{dataset_save_name}/{model_save_name}/')
        + '__'.join([
            f'{k}:{v}'
            for k, v in args.items()
            if k not in ['model_id', 'gpu_id', 'prefixes', 'dataset', 'history_len', 'hp_subset']
        ])
        + '.pkl'
    )

    print(f'completions save path: {completions_save_path}')

    # if output already exists, skip all inference steps
    try:
        completions = pickle.load(open(completions_save_path, 'rb'))
    except Exception:
        completions = []

    if len(completions) < len(dataset):
        # start the TGI container
        tgi_run = TGIRunner(
            model,
            args,
            TGI_IMAGE,
            os.path.abspath(os.path.join(LOCAL_ROOT_PATH, VOLUME.format(model=model_name))),
            hf_token=open(hf_token_path, 'r').read(),
            gpu_id=args['gpu_id'],
        )
        print('container up successfully for', model_name)

        # run dataset
        print(f'running inference on dataset {args["dataset"]}')
        tgi_run.run_dataset(
            dataset,
            save_path=completions_save_path,
            local_model_path=model_local_path,
            existing_completions=completions,
        )

        # stop the container
        client = docker.from_env()
        if tgi_run.container in client.containers.list():
            tgi_run.container.kill()
            time.sleep(30)  # wait for container to die
        print('container down successfully for', model_name)
