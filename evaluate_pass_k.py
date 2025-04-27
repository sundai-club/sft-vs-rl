import argparse
import json
import math
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams
import random

from countdown_verifier import extract_solution, validate_equation, evaluate_equation

def estimate_pass_k(num_samples: list[int], num_correct: list[int], k: int) -> float:
    """
    Estimates pass@k of each problem and returns the average pass@k over the dataset.
    num_samples: List where each element is the number of samples generated per problem.
    num_correct: List where each element is the number of correct solutions for that problem.
    k: The value of k for pass@k calculation.
    """
    if not isinstance(num_samples, list): num_samples = list(num_samples)
    if not isinstance(num_correct, list): num_correct = list(num_correct)

    def estimator(n: int, c: int, k: int) -> float:
        """Calculates 1 - C(n-c, k) / C(n, k)."""
        if n - c < k:
            return 1.0
        # Simplified calculation: prod_{i=0}^{k-1} (1 - (n-c-i)/(n-i))
        term = 1.0
        for i in range(k):
            term *= (n - c - i) / (n - i)
        return 1.0 - term

    pass_k_estimates = [estimator(int(n), int(c), k) for n, c in zip(num_samples, num_correct)]
    return np.mean(pass_k_estimates)


def main(args):
    try:
        df = pd.read_parquet(args.dataset_path)
        all_prompts = df['prompt'].apply(lambda x: x[0]['content']).tolist()
        all_ground_truths = df['reward_model'].apply(lambda x: x['ground_truth']).tolist()
        print(f"Loaded full dataset from {args.dataset_path}, shape: {df.shape}")
    except Exception as e:
        print(f"Error loading dataset with pandas: {e}")
        print("Attempting to load with datasets library...")
        try:
            from datasets import load_dataset
            dataset = load_dataset('parquet', data_files=args.dataset_path, split='train')
            all_prompts = [item['prompt'][0]['content'] for item in dataset]
            all_ground_truths = [item['reward_model']['ground_truth'] for item in dataset]
            print(f"Loaded full dataset with datasets library, {len(all_prompts)} samples.")
        except Exception as e2:
            print(f"Error loading dataset with datasets library: {e2}")
            return

    total_samples = len(all_prompts)
    if args.n_test is not None and 0 < args.n_test < total_samples:
        print(f"Randomly sampling {args.n_test} samples from the dataset (total: {total_samples}).")
        random.seed(args.random_seed) # for reproducibility
        sample_indices = random.sample(range(total_samples), args.n_test)
        prompts = [all_prompts[i] for i in sample_indices]
        ground_truths = [all_ground_truths[i] for i in sample_indices]
    else:
        if args.n_test is not None and args.n_test >= total_samples:
             print(f"n_test ({args.n_test}) >= total samples ({total_samples}). Using all samples.")
        elif args.n_test is not None:
             print(f"Invalid n_test value ({args.n_test}). Using all samples.")
        else:
             print("n_test not specified. Using all samples.")
        prompts = all_prompts
        ground_truths = all_ground_truths

    print(f"Running evaluation on {len(prompts)} samples.")

    llm = LLM(model=args.model_path, trust_remote_code=True, tensor_parallel_size=args.tensor_parallel_size)
    print(f"Initialized LLM: {args.model_path}")

    max_k = max(args.k_values)
    sampling_params = SamplingParams(
        n=max_k,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=["<|im_end|>", "<|endoftext|>"] 
    )
    print(f"Sampling params: n={max_k}, temp={args.temperature}, top_p={args.top_p}, max_tokens={args.max_tokens}, stop={sampling_params.stop}")

    print(f"Generating {max_k} solutions for each of the {len(prompts)} prompts...")
    outputs = llm.generate(prompts, sampling_params)
    print("Generation complete.")

    print("Evaluating generated solutions...")
    num_correct_per_prompt = []
    results_data = []

    for i, output in enumerate(tqdm(outputs)):
        prompt = output.prompt
        generated_samples = output.outputs
        target = int(ground_truths[i]['target']) # Cast to int
        numbers = [int(n) for n in ground_truths[i]['numbers']] # Cast each element to int
        
        correct_count = 0
        prompt_results = {'prompt_index': i, 'target': target, 'numbers': numbers, 'correct_solutions': [], 'incorrect_solutions': []}

        for sample in generated_samples:
            solution_text = sample.text
            equation = extract_solution(solution_text)
            is_correct = False
            
            if equation:
                valid_nums = validate_equation(equation, numbers)
                if valid_nums:
                    try:
                        result = evaluate_equation(equation)
                        if result is not None and abs(result - target) < 1e-5:
                            is_correct = True
                    except Exception:
                        pass # Evaluation error means incorrect

            if is_correct:
                correct_count += 1
                prompt_results['correct_solutions'].append(equation)
            else:
                prompt_results['incorrect_solutions'].append({'raw_output': solution_text, 'extracted_equation': equation})

        num_correct_per_prompt.append(correct_count)
        results_data.append(prompt_results)
        
    print("Evaluation complete.")
    
    if args.results_output_path:
        with open(args.results_output_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        print(f"Detailed results saved to {args.results_output_path}")

    print("\n--- Pass@k Results ---")
    num_samples_per_prompt = [max_k] * len(prompts)
    pass_k_results = []

    for k in args.k_values:
        if k > max_k:
            print(f"Warning: k={k} is greater than the number of generated samples ({max_k}). Skipping.")
            continue
        pass_k_value = estimate_pass_k(num_samples_per_prompt, num_correct_per_prompt, k)
        print(f"Pass@{k}: {pass_k_value:.4f}")
        pass_k_results.append({"k": k, "pass_k_accuracy": pass_k_value})

    # Save pass@k results
    if args.pass_k_output_path:
        try:
            pass_k_df = pd.DataFrame(pass_k_results)
            pass_k_df.to_csv(args.pass_k_output_path, index=False)
            print(f"Pass@k results saved to {args.pass_k_output_path}")
        except Exception as e:
            print(f"Error saving pass@k results to {args.pass_k_output_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model pass@k accuracy on the Countdown task.")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen1.5-1.8B-Chat", help="Path to the VLLM model (e.g., Hugging Face identifier).")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the Parquet dataset file.")
    parser.add_argument("--k_values", type=int, nargs='+', default=[1, 2, 8, 16, 32, 128], help="Values of k for pass@k calculation.")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature.")
    parser.add_argument("--top_p", type=float, default=0.95, help="Sampling top_p.")
    parser.add_argument("--max_tokens", type=int, default=256, help="Maximum number of tokens to generate.")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of GPUs for tensor parallelism.")
    parser.add_argument("--results_output_path", type=str, default=None, help="Optional path to save detailed evaluation results (JSON).")
    parser.add_argument("--pass_k_output_path", type=str, default=None, help="Optional path to save pass@k results (CSV).")
    parser.add_argument("--n_test", type=int, default=None, help="Number of samples to randomly select from the test set. If None or invalid, use all samples.")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for test set sampling.")
    
    args = parser.parse_args()

    args.k_values = sorted(list(set(args.k_values)))

    main(args)