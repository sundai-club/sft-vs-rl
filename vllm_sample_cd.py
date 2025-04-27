import os
import time
import sys
import numpy as np
import pandas as pd
import json
from optparse import OptionParser

from vllm import LLM, SamplingParams
from datasets import load_dataset

from transformers import AutoTokenizer

def parse_args():
    parser = OptionParser()
    
    parser.add_option("--idx_start", type="int", dest="idx_start")
    parser.add_option("--idx_end", type="int", dest="idx_end")
    
    (options, args) = parser.parse_args()

    return options

def load_prompts(parquet_file):
    dataset = load_dataset("parquet", data_files=parquet_file)
    prompts = [item[0]['content'] for item in dataset['train']['prompt']]
    return prompts

def save_append_jsonl(file_path, data):
    with open(file_path, "a") as f:
        for item in data:
            json.dump(item, f)
            f.write("\n")

def parse_and_save_responses(responses, file_path):
    all_responses = [[{'role': 'user', 'content': output.prompt},
                     {'role': 'assistant', 'content': output.outputs[0].text}]
                     for output in responses]

    save_append_jsonl(file_path, all_responses)

    return all_responses
    
def main(results_dir = './results'):

    options = parse_args()
    print(options)
    
    idx_start = options.idx_start
    idx_end = options.idx_end

    prompts = load_prompts("sft-vs-rl/train.parquet")
    prompts = prompts[idx_start:idx_end]
    
    model_name = 'Qwen/Qwen2.5-32B-Instruct'
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    llm = LLM(model=model_name, gpu_memory_utilization=0.95, tensor_parallel_size=1)

    sampling_params = SamplingParams(max_tokens = 2000,
                                 n = 1,
                                 temperature = 0.7)

    gens = llm.generate(prompts, sampling_params, use_tqdm=True)

    parse_and_save_responses(gens, f'{results_dir}/{idx_start}_{idx_end}.jsonl')

if __name__ == "__main__":
    main()