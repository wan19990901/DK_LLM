import argparse
from LLM_agent import LLM_agent
from Parsers import Base_Parser,CoT_Parser
import os
import pandas as pd
from tqdm import tqdm
import json
from typing import Dict, List, Any
import time
from collections import deque
from datetime import datetime, timedelta

DATA_DIR = '../data/'

# Experiment Config
NUM_OF_SAMPLES = 100
NUM_OF_REPEAT = 10

def get_llm_config(args) -> Dict[str, Any]:
    """Get LLM configuration from arguments"""
    return {
        'llm_type': args.llm_type,
        'api_key_link': args.api_key_file,
        'model': args.model,
        'prompt_link': args.prompt_file,
        'parser_template': CoT_Parser,  # We need to make this more flexible
        'temperature': args.temperature,
        'dataset': args.dataset,  # Add dataset to config
    }

def save_json(results: Dict[str, Any], llm_config: Dict[str, Any]) -> None:
    """Save results to a JSON file"""
    storage_dir = f'../data/results/{llm_config["model"]}'
    if not os.path.exists(storage_dir):
        os.makedirs(storage_dir)
    
    # Create filename using model, dataset name, and prompt file name
    prompt_name = os.path.splitext(os.path.basename(llm_config['prompt_link']))[0]  # Get prompt name without extension
    file_name = f'{llm_config["dataset"]}_{prompt_name}.json'
    file_path = os.path.join(storage_dir, file_name)
    
    # Add metadata
    results['metadata'] = {
        'prompt_file': os.path.basename(llm_config['prompt_link']),
        'num_samples': NUM_OF_SAMPLES,
        'llm_type': llm_config['llm_type'],
        'temperature': llm_config['temperature']
    }
    
    with open(file_path, 'w') as f:
        json.dump(results, f, indent=2)



class RateLimiter:
    def __init__(self, max_requests, time_window):
        self.max_requests = max_requests
        self.time_window = time_window  # in seconds
        self.requests = deque()

    def wait_if_needed(self):
        now = datetime.now()
        
        # Remove requests older than the time window
        while self.requests and (now - self.requests[0]) > timedelta(seconds=self.time_window):
            self.requests.popleft()
        
        # If we've hit the rate limit, wait until we can make another request
        if len(self.requests) >= self.max_requests:
            wait_time = (self.requests[0] + timedelta(seconds=self.time_window) - now).total_seconds()
            if wait_time > 0:
                print(f"\nRate limit reached. Waiting {wait_time:.2f} seconds...")
                time.sleep(wait_time)
                # Clean up old requests again after waiting
                while self.requests and (datetime.now() - self.requests[0]) > timedelta(seconds=self.time_window):
                    self.requests.popleft()
        
        # Add the new request
        self.requests.append(now)

def process_questions(llm_config: Dict[str, Any], start_index: int = 0) -> None:
    """Process questions and collect results"""
    MAX_PARSE_ATTEMPTS = 3  # Maximum number of attempts for parsing errors
    
    # Initialize rate limiter for Gemini
    rate_limiter = RateLimiter(max_requests=15, time_window=61) if llm_config['llm_type'] == 'gemini' else None
    
    with open(llm_config['api_key_link'], 'r') as f:
        api_key = f.read().strip()

    df = pd.read_csv(os.path.join(DATA_DIR, f'{llm_config["dataset"]}.csv'))
    df_subset = df[:NUM_OF_SAMPLES]

    results_dict = {
        'metadata': {
            'prompt_file': os.path.basename(llm_config['prompt_link']),
            'num_samples': NUM_OF_SAMPLES,
            'llm_type': llm_config['llm_type'],
            'temperature': llm_config['temperature']
        },
        'results': []
    }
    cot_agent = LLM_agent(
        llm_type=llm_config['llm_type'], 
        api_key=api_key, 
        model=llm_config['model'],
        temperature=llm_config['temperature']
    )
    # Process each question
    for row_idx in tqdm(range(start_index, len(df_subset)), colour='blue', desc='Questions', position=0):
        row = df_subset.iloc[row_idx]
        
        # Initialize agent once per question

        cot_agent.setup_prompt(llm_config['prompt_link'], llm_config['parser_template'])

        base_result = {
            'question_id': row_idx,
            'category': row['Category'],
            'name': row.get('Name', ''),
            'question_text': row['Question'],
            'correct_answer': row.get('Correct Answer', None),
            'model': llm_config['model']
        }

        # Multiple repeats for each question
        for repeat in tqdm(range(NUM_OF_REPEAT), colour='green', desc='Repeats', position=1):
            arguments_dict = {'question': row['Question']}
            
            # Apply rate limiting if using Gemini
            if rate_limiter:
                rate_limiter.wait_if_needed()
            
            # Try multiple times for each repeat
            success = False
            last_error = None
            parse_attempts = 0
            
            while not success and parse_attempts < MAX_PARSE_ATTEMPTS:
                try:
                    response = cot_agent.invoke(arguments_dict)
                    print(1)
                    print(response)
                    # Try to update result entry to verify response format
                    result_entry = base_result.copy()
                    result_entry.update({
                        'repeat_number': repeat,
                        'answer': response['Answer'],
                        'reasoning': response['reasoning'] # change this to confidence later
                    })
                    success = True
                except Exception as e:
                    last_error = str(e)
                    parse_attempts += 1
                    if parse_attempts < MAX_PARSE_ATTEMPTS:
                        print(f"Attempt {parse_attempts} failed: {last_error}, retrying...")
                        # Apply rate limiting for retries as well if using Gemini
                        if rate_limiter:
                            rate_limiter.wait_if_needed()
            
            # If we haven't successfully created a result entry, create one with error info
            if not success:
                result_entry = base_result.copy()
                result_entry.update({
                    'repeat_number': repeat,
                    'answer': f"Error after {MAX_PARSE_ATTEMPTS} attempts: {last_error}",
                    'reasoning': None
                })
            
            results_dict['results'].append(result_entry)

        # Save results after completing all repeats for the current question
        print(f"\nSaving results after completing question {row_idx}")
        save_json(results_dict, llm_config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run GSM8K evaluation')
    parser.add_argument('--llm_type', default='anthropic', 
                       choices=['openai', 'ollama', 'anthropic', 'gemini', 'azure', 'lambda'], 
                       help='LLM type')
    parser.add_argument('--api_key_file', default='../Api_keys/claude_api.txt', 
                       help='API key file')
    parser.add_argument('--model', default='claude-3-5-haiku-20241022', 
                       help='Model name')
    parser.add_argument('--prompt_file', default='../prompts/base.json', 
                       help='Prompt template file')
    parser.add_argument('--temperature', type=float, default=0.1, 
                       help='Temperature for LLM')
    parser.add_argument('--start_index', type=int, default=0, 
                       help='Starting index for processing')
    # Add new dataset argument
    parser.add_argument('--dataset', default='GSM8K', 
                       help='Dataset name (default: GSM8K)')
    args = parser.parse_args()
    llm_config = get_llm_config(args)
    process_questions(llm_config, args.start_index)
