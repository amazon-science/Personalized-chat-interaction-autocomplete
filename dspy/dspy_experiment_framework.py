# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

import argparse
import time

import pandas as pd
import phoenix as px
from litellm.exceptions import RateLimitError
from openinference.instrumentation.bedrock import BedrockInstrumentor
from openinference.instrumentation.dspy import DSPyInstrumentor
from phoenix.experiments import run_experiment
from phoenix.otel import register

import dspy

DATABASE_NAME = 'ranker_train_reasoning-full'
TRAIN_SIZE = 1
EXPERIMENT_PREFIX = 'personal-ranker-exp-reasoning'


# Define a dummy signature for testing
class DummySignature(dspy.Signature):
    """A simple signature that classifies text."""

    input_text = dspy.InputField(description='Input text to classify')
    label = dspy.OutputField(description='Classification label (positive/negative)')


# Implement dummy module
class DummyModule(dspy.Module):
    def __init__(self, use_labeled_examples=True):
        super().__init__()
        # Create few-shot examples
        self.examples = [
            dspy.Example(input_text='I love this!', label='positive'),
            dspy.Example(input_text='This is terrible.', label='negative'),
        ]
        self.classifier = dspy.ChainOfThought(DummySignature)

    def forward(self, input_text):
        result = self.classifier(input_text=input_text)
        return result.label


class DummyRankerModule(dspy.Module):
    def __init__(self):
        super().__init__()

    def forward(self, user_profile, prefix, suggestions):
        # Just return the suggestions in reverse order for testing
        return '5, 4, 3, 2, 1'


class LLMRankerSignature(dspy.Signature):
    """Rank suggestions by predicting what the user will most likely type next.

    Focus on TYPING LIKELIHOOD, not helpfulness or completeness. Common words like 'the', 'and', 'of', 'to', 'in' are often correct even if they seem generic. Partial completions are perfectly acceptable - prioritize what users actually type, not what would make the best sentence.

    Consider: What would this specific user naturally type next given their writing patterns and the context? Think about statistical frequency of words in natural language, not semantic richness."""

    user_profile = dspy.InputField(
        description='User context for understanding their typical language patterns and writing style.'
    )
    prefix = dspy.InputField(description='Text prefix typed by the user.')
    suggestions = dspy.InputField(description='List of 5 suggested completions, numbered 1–5.')
    reasoning = dspy.InputField(
        description='Explanation of ranking reasoning focusing on typing likelihood.', required=False
    )

    ranked_suggestions = dspy.OutputField(
        description="Ranked suggestion numbers from most to least likely to be typed next by this user, comma-separated. E.g., '1, 3, 2, 4, 5'. Prioritize statistical likelihood and natural typing patterns over semantic value."
    )


# Implement ranker module
class LLMRankerModule(dspy.Module):
    def __init__(self, use_labeled_examples=True):
        super().__init__()
        self.ranker = dspy.ChainOfThought(LLMRankerSignature)

    def forward(self, user_profile, prefix, suggestions, reasoning=None):
        # Pass reasoning if provided, otherwise don't include it in the kwargs
        kwargs = {
            'user_profile': user_profile,
            'prefix': prefix,
            'suggestions': suggestions,
        }
        if reasoning is not None:
            kwargs['reasoning'] = reasoning

        # Add retry mechanism for RateLimitError
        max_retries = 3  # Try up to 2 additional times (3 total attempts)
        retry_count = 0
        retry_delay = 5  # Initial delay in seconds

        while True:
            try:
                result = self.ranker(**kwargs)
                return result.ranked_suggestions
            except RateLimitError as e:
                retry_count += 1
                if retry_count > max_retries:
                    # If we've exceeded max retries, re-raise the exception
                    print(f'Failed after {retry_count} retries, raising exception')
                    raise

                print(
                    f'RateLimitError encountered: {str(e)}. Retrying in {retry_delay} seconds... (Attempt {retry_count}/{max_retries})'
                )
                time.sleep(retry_delay)
                # Exponential backoff for subsequent retries
                retry_delay *= 2


class MRRRankerMetric:
    def __init__(self):
        pass

    def __call__(self, gold, pred, reasoning=None):
        output_list = [item.strip() for item in pred.split(',')]
        expected_list = [item.strip() for item in gold['ranked_suggestions'].split(',')]

        for rank, item in enumerate(output_list, start=1):
            if item == expected_list[0]:
                return 1.0 / rank  # Return reciprocal of the first correct prediction

        return 0.0  # No correct predictions found


class DummyRankerMetric:
    def __init__(self):
        pass

    def __call__(self, gold, pred):
        return 0.5


class BalancedAccuracyMetric:
    def __init__(self, test_cases):
        # Calculate true ratio from test cases
        true_count = sum(1 for case in test_cases if case.label == 'positive')
        self.true_ratio = true_count / len(test_cases)

    def __call__(self, gold, pred):
        # Weight the score based on the true/false distribution
        if gold.label == 'positive':
            weight = 1.0 / (2 * self.true_ratio)
        else:
            weight = 1.0 / (2 * (1 - self.true_ratio))

        return (pred == gold.label) * weight


def init_tracing():
    try:
        endpoint = 'http://localhost:4317'
        tracer_provider = register(project_name='dspy_experiments', endpoint=endpoint, set_global_tracer_provider=False)

        # Check for required modules
        import importlib

        required_modules = ['dsp', 'openinference']
        for mod in required_modules:
            try:
                importlib.import_module(mod)
            except ImportError:
                print(f"Warning: Required module '{mod}' not found. Tracing may not work properly.")

        DSPyInstrumentor().instrument(tracer_provider=tracer_provider, skip_dep_check=True)
        BedrockInstrumentor().instrument(tracer_provider=tracer_provider)
        print('Tracing initialized successfully')
    except Exception as e:
        print(f'Error in init_tracing: {e}')
        raise


def evaluate_with_phoenix(
    module, devset_df: pd.DataFrame, experiment_name: str = 'dspy-experiment', dataset_name: str = 'dspy_evaluation'
):
    """Evaluate the model using Phoenix experiments"""
    print('Evaluating with Phoenix experiments...')

    if 'suggestions' not in devset_df.columns:
        if 'completions' in devset_df.columns:
            devset_df['suggestions'] = devset_df['completions']
        else:
            devset_df['suggestions'] = devset_df['shuffled_completions']

    # Ensure devset_df is not empty
    if devset_df.empty:
        print('Warning: Empty evaluation dataset')
        return None, None

    # # Calculate true ratio for balanced accuracy
    # try:
    #     true_count = sum(1 for _, row in devset_df.iterrows() if row['label'] == "positive")
    #     true_ratio = true_count / len(devset_df) if len(devset_df) > 0 else 0.5  # Default to 0.5 if empty
    # except Exception as e:
    #     print(f"Error calculating true ratio: {e}")
    #     true_ratio = 0.5  # Default if there's an error

    # Try to get existing dataset, create if not found
    try:
        client = px.Client()
        try:
            dataset = client.get_dataset(name=dataset_name)
            print(f'Using existing dataset: {dataset_name}')
        except (ValueError, Exception) as e:
            print(f'Creating new dataset: {dataset_name}, reason: {e}')
            dataset = client.upload_dataset(
                dataset_name=dataset_name,
                dataframe=devset_df,
                input_keys=['user_profile', 'prefix', 'suggestions'],
                output_keys=['ranked_suggestions'],
            )
    except Exception as e:
        print(f'Error accessing Phoenix: {e}')
        return None, None

    # # Define task for Phoenix experiment
    # def classification_task(input) -> str:
    #     # Phoenix sends inputs as a dictionary with the field names
    #     if isinstance(input, dict) and "input_text" in input:
    #         return module(input["input_text"])
    #     # Handle direct string inputs
    #     return module(input)

    # Define task for Phoenix experiment
    def ranking_task(input) -> str:
        return module(user_profile=input['user_profile'], prefix=input['prefix'], suggestions=input['suggestions'])

    # # Create evaluators
    # def balanced_accuracy_evaluator(output: str, expected: dict, true_ratio: float) -> float:
    #     weight = 1.0 / (2 * true_ratio) if expected["label"] == "positive" else 1.0 / (2 * (1 - true_ratio))
    #     return float(output == expected["label"]) * weight

    # def accuracy_evaluator(output: str, expected: dict) -> float:
    #     return float(output == expected["label"])

    def mrr_evaluator(output: str, expected: dict) -> float:
        output_list = [item.strip() for item in output.split(',')]
        expected_list = [item.strip() for item in expected['ranked_suggestions'].split(',')]

        for rank, item in enumerate(output_list, start=1):
            if item == expected_list[0]:
                return 1.0 / rank  # Return reciprocal of the first correct prediction

        return 0.0  # No correct predictions found

    # # Create evaluator with true_ratio closure
    # balanced_acc = partial(balanced_accuracy_evaluator, true_ratio=true_ratio)
    # balanced_acc.__name__ = "balanced_accuracy"

    # Run experiment with error handling
    try:
        experiment_results = run_experiment(
            dataset,
            ranking_task,
            experiment_name=experiment_name,
            experiment_description='Evaluation of completions ranker model',
            evaluators=[mrr_evaluator],
            experiment_metadata={'model': 'ranker'},
        )

        return experiment_results.eval_runs[0].result
        # return experiment_results.mean_scores["balanced_accuracy"], experiment_results.mean_scores["accuracy"]
    except Exception as e:
        print(f'Error running experiment {experiment_name}: {e}')
        return None, None


def run_optimization_experiments(
    train_path: str,
    test_cases_path: str,
    experiment_prefix: str = 'dspy-experiment',
    dataset_name: str = 'dspy_evaluation',
    train_size: int = 20,
):
    """Run experiments with different DSPy optimizers and log results with Phoenix"""
    print('Running optimization experiments...')

    # Read and prepare train data
    train_df = pd.read_csv(train_path, index_col=0)
    train_cases = [
        dspy.Example(
            user_profile=row['user_profile'],
            prefix=row['prefix'],
            suggestions=row['shuffled_completions'],
            reasoning=row['reasoning'],
            ranked_suggestions=row['ranked_suggestions'],
        ).with_inputs('user_profile', 'prefix', 'suggestions')
        for _, row in train_df.iterrows()
    ]

    # Read and prepare test data
    test_df = pd.read_csv(test_cases_path, index_col=0)
    test_cases = [
        dspy.Example(
            user_profile=row['user_profile'],
            prefix=row['prefix'],
            suggestions=row['completions'],
            ranked_suggestions=row['ranked_suggestions'],
        ).with_inputs('user_profile', 'prefix', 'suggestions')
        for _, row in test_df.iterrows()
    ]

    # Initialize metric
    metric = MRRRankerMetric()

    # Split into train/dev sets
    trainset = train_cases
    devset = test_cases
    # devset = trainset

    # Create devset dataframe for Phoenix evaluation
    devset_df = test_df
    # devset_df = train_df

    # Define experiments
    experiments = {
        'zero_shot': {'optimizer': None, 'config_path': 'zero_shot_config.json', 'needs_validation': False},
        'labeled_few_shot': {
            # "optimizer": dspy.LabeledFewShot(k=5),
            'optimizer': dspy.LabeledFewShot(k=3),
            'config_path': 'labeled_few_shot_config.json',
            'needs_validation': False,
        },
        'bootstrap_rs': {
            'optimizer': dspy.BootstrapFewShotWithRandomSearch(
                metric=metric, max_bootstrapped_demos=4, max_labeled_demos=16, num_candidate_programs=8, max_rounds=1
            ),
            'config_path': 'bootstrap_rs_config.json',
            'needs_validation': True,
        },
        'mipro_v2': {
            'optimizer': dspy.MIPROv2(
                metric=metric,
                # # max_bootstrapped_demos=4,
                max_bootstrapped_demos=2,
                # # max_labeled_demos=16,
                max_labeled_demos=4,
                # auto=None,  # Set auto to None to allow specifying num_candidates
                # num_trials=5,
                # num_candidates=8
            ),
            'config_path': 'mipro_v2_config.json',
            'needs_validation': True,
        },
    }

    results = {}

    # Run each experiment
    for exp_name, exp_config in experiments.items():
        print(f'\nRunning {exp_name} experiment...')

        try:
            # Create fresh module instance
            module = LLMRankerModule()
            # ctrainset = [v.with_inputs('input_text') for v in module.examples]
            # if exp_name != "labeled_few_shot":
            #     ctrainset = trainset + ctrainset
            ctrainset = trainset

            if exp_config['optimizer'] is not None:
                try:
                    # Compile with optimizer, handling different interfaces
                    if exp_config['needs_validation']:
                        compile_kwargs = {'requires_permission_to_run': False} if exp_name == 'mipro_v2' else {}
                        module = exp_config['optimizer'].compile(
                            module,
                            # num_trials=8,
                            trainset=ctrainset,
                            valset=devset,
                            # valset=ctrainset,
                            **compile_kwargs,
                        )
                    else:
                        module = exp_config['optimizer'].compile(module, trainset=ctrainset)

                    # Save optimized configuration
                    try:
                        module.save(exp_config['config_path'])
                        print(f'Saved configuration to {exp_config["config_path"]}')
                    except Exception as e:
                        print(f'Error saving configuration: {e}')
                except Exception as e:
                    print(f'Error during compilation with {exp_name} optimizer: {e}')
                    # Continue with the unoptimized module
                    print('Continuing with unoptimized module')

            # Evaluate using Phoenix
            experiment_name = f'{experiment_prefix}-{exp_name}'
            results[exp_name] = evaluate_with_phoenix(
                module,
                devset_df,
                # train_df,
                experiment_name=experiment_name,
                dataset_name=dataset_name,
            )
        except Exception as e:
            print(f'Error in experiment {exp_name}: {e}')
            results[exp_name] = None

    # Print summary
    print('\nExperiment Results Summary:')
    print('-' * 50)
    for exp_name, result in results.items():
        print(f'\n{exp_name.upper()}:')
        print(f'Configuration saved to: {experiments[exp_name]["config_path"]}')
        print(f'Experiment name: {experiment_prefix}-{exp_name}')
        print(f'Score: {result.score}')
        # if result and result[0] is not None and result[1] is not None:
        #     balanced_acc, regular_acc = result
        #     print(f"Balanced Accuracy: {balanced_acc.score:.3f}")
        #     print(f"Regular Accuracy: {regular_acc.score:.3f}")
        # else:
        #     print("Evaluation failed")

    return results


def configure_aws_credentials(aws_profile=None, aws_region=None, aws_key_id=None, aws_secret_key=None):
    """Configure AWS credentials for Bedrock"""
    import os

    import boto3

    print('Configuring AWS credentials for Bedrock...')

    # Set region from parameter, environment variable, or default
    region = aws_region or os.environ.get('AWS_REGION', 'us-west-2')

    # Set region in environment
    os.environ['AWS_REGION'] = region
    print(f'Setting AWS_REGION to {region}')

    # Check for direct credentials passed as arguments
    if aws_key_id and aws_secret_key:
        print('Using AWS credentials provided via command line arguments')
        os.environ['AWS_ACCESS_KEY_ID'] = aws_key_id
        os.environ['AWS_SECRET_ACCESS_KEY'] = aws_secret_key
        return True

    # Check for credentials in environment variables
    elif os.environ.get('AWS_ACCESS_KEY_ID') and os.environ.get('AWS_SECRET_ACCESS_KEY'):
        print('Using AWS credentials from environment variables')
        return True

    # Configure AWS session with profile if specified
    elif aws_profile:
        try:
            print(f'Using AWS profile: {aws_profile}')
            session_kwargs = {'profile_name': aws_profile}

            # Create session with profile
            session = boto3.Session(**session_kwargs)

            # Check if credentials are available
            credentials = session.get_credentials()
            if credentials is None:
                print(f"No credentials found in profile '{aws_profile}'")
                return False

            # Set credentials as environment variables (for libraries that use them)
            frozen_creds = credentials.get_frozen_credentials()
            os.environ['AWS_ACCESS_KEY_ID'] = frozen_creds.access_key
            os.environ['AWS_SECRET_ACCESS_KEY'] = frozen_creds.secret_key
            if frozen_creds.token:
                os.environ['AWS_SESSION_TOKEN'] = frozen_creds.token

            print('AWS credentials configured successfully using profile')
            return True
        except Exception as e:
            print(f'Error configuring AWS credentials from profile: {e}')
            return False

    # Try default session as a last resort
    else:
        try:
            session = boto3.Session()
            credentials = session.get_credentials()

            if credentials is None:
                print('No AWS credentials found')
                return False

            # Set credentials as environment variables
            frozen_creds = credentials.get_frozen_credentials()
            os.environ['AWS_ACCESS_KEY_ID'] = frozen_creds.access_key
            os.environ['AWS_SECRET_ACCESS_KEY'] = frozen_creds.secret_key
            if frozen_creds.token:
                os.environ['AWS_SESSION_TOKEN'] = frozen_creds.token

            print('AWS credentials found in default configuration')
            return True
        except Exception as e:
            print(f'Error configuring default AWS credentials: {e}')
            return False


def main():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--test-path',
            default='./dspy/val.csv',
            help='Path to test cases CSV file',
        )
        parser.add_argument(
            '--train-path',
            default='./dspy/train_reasoning.csv',
            help='Path to train set CSV file',
        )
        parser.add_argument('--no-tracing', default=False, action='store_true', help='Disable tracing')
        parser.add_argument('--experiment-prefix', default=EXPERIMENT_PREFIX, help='Name for the Phoenix experiment')
        parser.add_argument('--dataset-name', default=DATABASE_NAME, help='Name for the Phoenix dataset')
        parser.add_argument('--train-size', type=int, default=TRAIN_SIZE, help='Number of examples to use for training')
        parser.add_argument(
            '--aws-profile', default='cline-profile', help='AWS profile name to use for Bedrock authentication'
        )
        parser.add_argument('--aws-region', default='us-west-2', help='AWS region for Bedrock')
        parser.add_argument('--aws-key-id', help='AWS access key ID for Bedrock authentication')
        parser.add_argument('--aws-secret-key', help='AWS secret access key for Bedrock authentication')
        args = parser.parse_args()

        print(f'Command line arguments: {vars(args)}')

        # Configure AWS credentials
        credentials_configured = configure_aws_credentials(
            args.aws_profile, args.aws_region, args.aws_key_id, args.aws_secret_key
        )

        model_str = 'us.anthropic.claude-3-7-sonnet-20250219-v1:0'
        print(f'Initializing DSPy with {model_str}')
        if credentials_configured:
            lm = dspy.LM(model_str, temperature=0.7)
            # lm = dspy.LM('bedrock/anthropic.claude-3-5-haiku-20241022-v1:0', temperature=0.7)
            # Test if model works
            try:
                print('Successfully connected to Bedrock API')
            except Exception as e:
                print(f'Error testing Bedrock connection: {e}')
                raise e
        else:
            raise Exception(
                'AWS credentials not properly configured. Please set up your AWS credentials by either:\n'
                '1. Using the --aws-profile option\n'
                '2. Setting AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables\n'
                '3. Configuring ~/.aws/credentials file'
            )

        dspy.configure(lm=lm)

        print(f'Command line arguments: {vars(args)}')

        if not args.no_tracing:
            print('Initializing tracing...')
            try:
                init_tracing()
                print('Tracing initialized successfully')
            except Exception as e:
                print(f'Warning: Failed to initialize tracing: {e}')
                print('Continuing without tracing...')

        print(f'Starting experiment with dataset: {args.test_path}')
        results = run_optimization_experiments(
            args.train_path,
            args.test_path,
            args.experiment_prefix,  # Correct parameter name
            args.dataset_name,
            args.train_size,
        )

        print('\nExperiment completed successfully!')
        return results

    except Exception as e:
        print(f'\n{"=" * 50}')
        print(f'ERROR: Experiment failed with error: {e}')
        import traceback

        print('\nTraceback:')
        traceback.print_exc()
        print(f'{"=" * 50}')
        return None


if __name__ == '__main__':
    main()
