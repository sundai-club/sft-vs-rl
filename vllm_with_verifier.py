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
import countdown_verifier

def parse_args():
    parser = OptionParser()
    
    parser.add_option("--idx_start", type="int", dest="idx_start")
    parser.add_option("--idx_end", type="int", dest="idx_end")
    parser.add_option("--max_tries", type="int", dest="max_tries", default=3)
    parser.add_option("--num_samples", type="int", dest="num_samples", default=16)
    
    (options, args) = parser.parse_args()

    return options

def load_messages(parquet_file):
    dataset = load_dataset("parquet", data_files=parquet_file)
    messages_list = dataset['train']['messages']
    targets = dataset['train']['target']
    numbers = dataset['train']['numbers']
    ground_truths = [{'target': target, 'numbers': numbers} for target, numbers in zip(targets, numbers)]
    return messages_list, ground_truths

def save_append_jsonl(file_path, data):
    with open(file_path, "a") as f:
        for item in data:
            json.dump(item, f)
            f.write("\n")

def main(results_dir='./results'):
    options = parse_args()
    print(options)
    
    idx_start = options.idx_start
    idx_end = options.idx_end
    max_tries = options.max_tries
    num_samples = options.num_samples

    os.makedirs(results_dir, exist_ok=True)
    
    messages_list, ground_truths = load_messages("sft-vs-rl/train.parquet")
    messages_subset = messages_list[idx_start:idx_end]
    
    ground_truths_subset = None
    if ground_truths is not None:
        ground_truths_subset = ground_truths[idx_start:idx_end]
    
    model_name = 'Qwen/Qwen2.5-32B-Instruct'
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    llm = LLM(model=model_name, gpu_memory_utilization=0.95, tensor_parallel_size=1)

    sampling_params = SamplingParams(
        max_tokens=2000,
        n=num_samples, 
        temperature=0.7
    )
    
    successful_indices = set()
    successful_responses = []
    remaining_indices = list(range(len(messages_subset)))
    
    for attempt in range(max_tries):
        print(f"Attempt {attempt+1}/{max_tries}")
        
        if not remaining_indices:
            print("All responses successful. Exiting.")
            break
        
        prompts = []
        for idx in remaining_indices:
            prompt = tokenizer.apply_chat_template(messages_subset[idx], tokenize=False)
            prompts.append(prompt)
        
        print(f"Generating {len(prompts)} prompts with {num_samples} samples each...")
        gens = llm.generate(prompts, sampling_params, use_tqdm=True)
        
        new_remaining_indices = []
        
        for i, gen in enumerate(gens):
            idx = remaining_indices[i]
            ground_truth = ground_truths_subset[idx] if ground_truths_subset is not None else None
            
            found_correct = False
            for j, output in enumerate(gen.outputs):
                response_text = output.text
                
                if ground_truth is None:
                    continue
                
                score = countdown_verifier.compute_score(
                    solution_str=response_text,
                    ground_truth=ground_truth,
                    method='strict',
                    format_score=0.1,
                    score=1.
                )
                
                if score == 1.0:
                    successful_indices.add(idx)
                    conversation = [
                        {'role': 'user', 'content': gen.prompt},
                        {'role': 'assistant', 'content': response_text}
                    ]
                    successful_responses.append(conversation)
                    print(f"Item {idx_start + idx} correct! (Sample {j+1}/{num_samples})")
                    found_correct = True
                    break
            
            if not found_correct:
                new_remaining_indices.append(idx)
        
        remaining_indices = new_remaining_indices
    
    save_append_jsonl(f'{results_dir}/{idx_start}_{idx_end}.jsonl', successful_responses)
    

if __name__ == "__main__":
    main()