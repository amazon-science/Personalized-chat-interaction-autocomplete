Before running experiments, get the needed packages by running:
```
pip install -r requirements.txt
```

Experiments are run in 2 stages:

**1. Inference** - `run_baselines.py`
Generates completions for all prefixes in the dataset for a single model. Output is a completions `.pkl` file saved in `./completions`.
Important arguments (full list of arguments can be found in the file): 
- `model_id`: any huggingface model. If you have a finetuned model, try to give it a meaningful name that contains the word "finetuned", and then specify a path to S3 for your finetuned model in the code before `prepare_model` is called.
- `dataset`: currently supported datasets appear in the `choices` field of this argument. Your chosen dataset have a preperation file, such as `src/prepare_prism.py` and `src/prepare_wildchat.py`.
- `gpu_id`: to simultaneously run several models on different GPUs of the same instance, each `gpu_id` corresponds to a different port.
- Inference arguments: `best_of`, `max_new_tokens`, `top_n_tokens`, `temperature`, `top_p`. 

Make sure to put your huggingface token in `./resources/hf_token.txt`. 

Example:
```bash
python run_baselines.py --dataset wildchat --model_id mistralai/Mistral-7B-v0.1 --gpu_id 0 --best_of 5 --temperature 1.0
```


**2. Metrics** - `metrics.py`
Computes metrics such as `saved@k` and `acceptance_rate@k`. Outputs a `.csv` file saved in `./results/saved_at_k`.
Important arguments (full list of arguments can be found in the file): 
- `model_id`: if None, run for all models in `models_list`. You can specify a single model to run the metrics on.
- `dataset`: used to find the saved completions path, so make sure to use the appropriate name according to the dataset inference was run on.
- `rank_by`: confidence measure to rank completions by, e.g.: perplexity, entropty etc. Currently `log_likelihood` is the best ranker for all models and datasets, so use it as default if you're not experimenting.

Example:
```bash
python src/metrics.py --dataset wildchat --model_id mistralai/Mistral-7B-v0.1 --rank_by log_likelihood --personalization_scheme recency_turns --personalization_r 20
```
There are other arguments referring to previous experiments, such as word/char prefixes, single word/partial completions etc. The default values of these arguments are the ones we ran our baseline experiments with. 
