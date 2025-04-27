import countdown_verifier
import os
import pandas as pd

from vllm import LLM, SamplingParams

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
PROMPT_BATCH_SIZE = 32
NUM_PASSES = 256

def generate_batch(llm_prompts):
    # wrap the sync calls in threads to avoid blocking the event loop
    model_name = 'Qwen/Qwen2.5-1.5B-Instruct'
    llm = LLM(
        model=model_name, 
        gpu_memory_utilization=0.95, 
        max_model_len=2048,
        tensor_parallel_size=1, 
        enable_prefix_caching=True
    )
    sampling_params = SamplingParams(
        max_tokens = 2000,
        n=NUM_PASSES, # Change for best of 256 eval
        temperature = 0.7
    )
    responses = llm.generate(llm_prompts, sampling_params, use_tqdm=True)
    return [[output.text for output in response.outputs]for response in responses]

def generate_prompts(df):
    prompts = []
    for index, row in df.iterrows():
        target = row['target']
        numbers = row['nums']
        prompt = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
User: Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.
Assistant: Let me solve this step by step.
<think>"""
        prompts.append(prompt)
    return prompts

def main(filename='train.parquet'):
    # Load your DataFrame here
    df = pd.read_parquet(filename)

    START_IDX = 0
    END_IDX = 16

    df = df.iloc[START_IDX:END_IDX]
    
    df['nums'] = df['nums'].apply(lambda x: [int(num) for num in x])
    
    DATASET_SIZE = len(df)
    
    llm_prompts = generate_prompts(df)
    scores = [[None for _ in range(NUM_PASSES)] for _ in range(DATASET_SIZE)]
    for i in range(0, len(llm_prompts), PROMPT_BATCH_SIZE):
        batch_prompts = llm_prompts[i:i + PROMPT_BATCH_SIZE]
        responses = generate_batch(batch_prompts)
        flattened_responses = [item for sublist in responses for item in sublist]
        
        ground_truth = df[['target', 'nums']].to_dict(orient='records')
        
        for idx, response in enumerate(flattened_responses):
            row_idx = i + idx // NUM_PASSES
            col_idx = idx % NUM_PASSES
            
            solution_str = response
            target = ground_truth[row_idx]['target']
            numbers = ground_truth[row_idx]['nums']
            
            score = countdown_verifier.compute_score(
                solution_str=solution_str,
                ground_truth={'target': target, 'numbers': numbers},
                method='strict',
                format_score=0.1,
                score=1.
            )
            
            scores[row_idx][col_idx] = score
        
        # Save the scores to a file
        scores_df = pd.DataFrame(scores)
        scores_df.to_parquet('scores.parquet', index=False)
        
        pass_accuracy = (scores_df.sum(axis=1) > 0.1).mean()
        print(f"Pass accuracy: {pass_accuracy:.4f}")
        
        
if __name__ == "__main__":
    main()