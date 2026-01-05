# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0


import os

import pandas as pd

import dspy


class DSPyRankerForPrism:
    def __init__(self, root_path='./'):
        self.root_path = root_path
        self.user_profiles = {}
        self.dspy_module = None
        self.load_user_profiles()
        self.init_dspy_ranker()

    def load_user_profiles(self):
        """Load user profiles from validation data for mapping"""
        val_path = os.path.join(self.root_path, 'dspy', 'val.csv')
        if not os.path.exists(val_path):
            raise FileNotFoundError(
                f'DSPy validation data not found at {val_path}. Cannot proceed without user profiles.'
            )

        try:
            val_df = pd.read_csv(val_path, index_col=0)

            # Create mapping from user_id to user_profile (each user profile stored only once)
            for _, row in val_df.iterrows():
                user_id = row['user_id']
                user_profile = row['user_profile']

                # Store each user's profile only once since it remains constant for all prefixes
                if user_id not in self.user_profiles:
                    self.user_profiles[user_id] = user_profile

            if len(self.user_profiles) == 0:
                raise ValueError('No user profiles loaded from validation data')

            print(f'Successfully loaded profiles for {len(self.user_profiles)} unique users from DSPy validation data')

        except Exception as e:
            raise RuntimeError(f'Failed to load user profiles: {e}')

    def get_user_profile(self, user_id, prefix):
        """Get user profile for a given user_id (prefix not needed since profile is constant per user)"""
        if user_id not in self.user_profiles:
            raise KeyError(
                f'User ID {user_id} not found in DSPy validation data. Cannot provide personalized ranking without user profile.'
            )

        # Simply return the user's profile since it's constant for all prefixes
        return self.user_profiles[user_id]

    def init_dspy_ranker(self):
        """Initialize DSPy ranker with MIPRO-optimized configuration"""
        config_path = os.path.join(self.root_path, 'dspy', 'mipro_v2_config.json')
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f'MIPRO-optimized DSPy configuration not found at {config_path}. Run MIPRO optimization first.'
            )

        try:
            # Add DSPy directory to path first
            import sys

            dspy_dir = os.path.join(self.root_path, 'dspy')
            if dspy_dir not in sys.path:
                sys.path.append(dspy_dir)

            # Configure AWS credentials and LM exactly as used in MIPRO training
            from dspy_experiment_framework import LLMRankerModule, configure_aws_credentials

            # Configure AWS credentials (same as used in original experiment)
            credentials_configured = configure_aws_credentials(aws_profile='cline-profile', aws_region='us-west-2')

            if not credentials_configured:
                raise RuntimeError('AWS credentials not properly configured. DSPy ranker requires AWS Bedrock access.')

            # Configure DSPy with the exact same LM used in MIPRO training
            # model_str = 'bedrock/anthropic.claude-3-5-haiku-20241022-v1:0'
            model_str = 'us.anthropic.claude-3-7-sonnet-20250219-v1:0'
            lm = dspy.LM(model_str, temperature=0.7)

            # Test the LM connection
            try:
                print('Successfully connected to Bedrock API')
            except Exception as e:
                raise RuntimeError(f'Failed to connect to Bedrock API: {e}')

            dspy.configure(lm=lm)

            # Create module and load MIPRO-optimized configuration
            self.dspy_module = LLMRankerModule()
            self.dspy_module.load(config_path)

            print('Successfully loaded MIPRO-optimized DSPy ranker configuration')
            print('Using optimized prompt with 4 few-shot examples and enhanced instructions')
            print(f'Configured with {model_str}')
            self.dspy_available = True

        except Exception as e:
            raise RuntimeError(f'Failed to initialize MIPRO-optimized DSPy ranker: {e}')

    def rank_completions_dspy(self, user_profile, prefix, suggestions_text):
        """Use the MIPRO-optimized DSPy module to rank completions"""
        if not self.dspy_available or self.dspy_module is None:
            raise RuntimeError('MIPRO-optimized DSPy ranker not properly initialized')

        try:
            # Use the MIPRO-optimized DSPy module
            ranking = self.dspy_module(user_profile=user_profile, prefix=prefix, suggestions=suggestions_text)
            return ranking
        except Exception as e:
            raise RuntimeError(f'Error calling MIPRO-optimized DSPy module: {e}')

    def parse_ranking_output(self, ranking_str, completions_list):
        """Parse DSPy ranking output and assign confidence scores"""
        try:
            # Parse ranking like "3, 1, 2, 4, 5"
            ranking_order = [int(x.strip()) for x in ranking_str.split(',')]

            # Handle incomplete rankings by appending missing numbers
            if len(ranking_order) != len(completions_list):
                expected_numbers = set(range(1, len(completions_list) + 1))
                provided_numbers = set(ranking_order)
                missing_numbers = expected_numbers - provided_numbers

                # Append missing numbers to the end of the ranking
                ranking_order.extend(sorted(missing_numbers))

                print(f'Warning: Incomplete ranking detected. Original: {ranking_str}')
                print(f'Extended ranking: {ranking_order}')

            # Validate that we now have the correct numbers
            if set(ranking_order) != set(range(1, len(completions_list) + 1)):
                raise ValueError(
                    f'Invalid ranking order: {ranking_order}. Should contain exactly numbers 1-{len(completions_list)}'
                )

            # Assign confidence scores (higher rank = higher confidence)
            confidence_scores = {}
            max_score = len(ranking_order)

            for position, completion_idx in enumerate(ranking_order):
                completion = completions_list[completion_idx - 1]
                # Higher position gets lower score (position 0 gets highest score)
                confidence = max_score - position
                confidence_scores[completion] = confidence

            return confidence_scores

        except Exception as e:
            raise ValueError(f"Failed to parse ranking output '{ranking_str}': {e}")

    def rank(self, prefix, completions, user_id, confidence_measure_name):
        """Main ranking function using MIPRO-optimized DSPy ranker"""
        try:
            if len(completions) == 0:
                return completions

            # Completions are already filtered to top 5 by log-likelihood from ranker.py
            # DSPy MIPRO model expects exactly 5, so pad with empty strings if needed
            completions_list = list(completions.keys())
            while len(completions_list) < 5:
                completions_list.append('')

            # Format completions as numbered suggestions for MIPRO-optimized DSPy
            suggestions_text = '\n'.join([f'{i + 1}. {comp}' for i, comp in enumerate(completions_list)])

            # Get user profile (will raise exception if not found)
            user_profile = self.get_user_profile(user_id, prefix)

            # Get DSPy ranking using MIPRO-optimized model
            ranking_output = self.rank_completions_dspy(user_profile, prefix, suggestions_text)

            # Parse ranking and assign confidence scores
            confidence_scores = self.parse_ranking_output(ranking_output, completions_list)

            # Update completions with DSPy scores (only for non-empty completions)
            for completion, tokens_dict in completions.items():
                if completion in confidence_scores and completion != '':
                    tokens_dict[confidence_measure_name] = confidence_scores[completion]
                else:
                    # This shouldn't happen if everything is working correctly
                    tokens_dict[confidence_measure_name] = 0.0

            return completions

        except Exception as e:
            # Re-raise with context about where the error occurred
            raise RuntimeError(
                f"MIPRO-optimized DSPy ranking failed for user_id={user_id}, prefix='{prefix[:50]}...': {e}"
            )
