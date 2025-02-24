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

from parsers import GapFillingNumberParser
from parsers import BinaryChoiceParser
from parsers import MultipleChoiceParser
from utils import parse_correct_answer
from QuestionType import QuestionType

DATA_DIR = '../data/'

# Experiment Config
NUM_OF_SAMPLES = 2000
NUM_OF_REPEAT = 15

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

    report_properties = {
        "question_number": len(df_subset),
        "total_answers": len(df_subset) * NUM_OF_REPEAT,
        "successfully_parsed": 0,
        "is_correct_count": 0
    }

    # Process each question
    for row_idx in tqdm(range(start_index, len(df_subset)), colour='blue', desc='Questions', position=0):
        row = df_subset.iloc[row_idx]
        
        correct_answer = row.get('Correct Answer', None)
        question_type = parse_correct_answer(correct_answer)
        if (question_type != QuestionType.MULTIPLE_CHOICE):
            continue

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
            correct_answer = base_result['correct_answer']
            question_type = parse_correct_answer(correct_answer)

            if (question_type == QuestionType.UNDEFINED):
                base_result.update({
                    "fail_reason": f"Unable to identify question type for {correct_answer}"
                })
                break

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
                    # Try to update result entry to verify response format
                    result_entry = base_result.copy()
                    result_entry.update({
                        'repeat_number': repeat,
                        'answer': response['Answer'],
                        'reasoning': response['reasoning'] # Change this later
                    })

                    try:
                        match question_type:
                            case QuestionType.GAP_FILLING_NUMBER:
                                parsed_answer = GapFillingNumberParser.parse_answer(result_entry["answer"])
                                is_correct = GapFillingNumberParser.compare_answer(parsed_answer, correct_answer)
                                result_entry.update({
                                    "parsed_answer": parsed_answer,
                                    "is_correct": is_correct
                                })
                                report_properties.update({"parsed_number": report_properties["parsed_number"]+1})
                            
                            case QuestionType.BINARY_CHOICE:
                                parsed_answer = BinaryChoiceParser.parse_answer(result_entry["answer"])
                                is_correct = BinaryChoiceParser.compare_answer(parsed_answer, correct_answer)
                                result_entry.update({
                                    "parsed_answer": parsed_answer,
                                    "is_correct": is_correct
                                })
                                report_properties.update({"parsed_number": report_properties["parsed_number"]+1})

                            case QuestionType.MULTIPLE_CHOICE:
                                parsed_answer = MultipleChoiceParser.parse_answer(result_entry["answer"])
                                is_correct = MultipleChoiceParser.compare_answer(parsed_answer, correct_answer, base_result['question_text'])
                                result_entry.update({
                                    "parsed_answer": parsed_answer,
                                    "is_correct": is_correct
                                })
                                report_properties.update({"parsed_number": report_properties["parsed_number"]+1})
                    except Exception as e:
                        result_entry.update({
                            "fail_reason": str(e)
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
                    else:
                        result_entry.update({
                            "fail_reason": f"Error occur when invoking agent: {last_error}"
                        })
            
            # If we haven't successfully created a result entry, create one with error info
            if not success:
                result_entry = base_result.copy()
                result_entry.update({
                    'repeat_number': repeat,
                    'answer': f"Error after {MAX_PARSE_ATTEMPTS} attempts: {last_error}",
                    'reasoning': None # Need to make this more flexible; 
                })
            
            if "is_correct" not in result_entry:
                result_entry.update({"is_correct": None})

            if result_entry["is_correct"]:
                report_properties.update({"is_correct_count": report_properties["is_correct_count"]+1})

            results_dict['results'].append(result_entry)
        print(results_dict)
        print(llm_config)
        save_json(results_dict, llm_config)
        # Save results after completing all repeats for the current question
    print(f"\nSaving results after completing question {row_idx}")
    save_json(results_dict, llm_config)
    print(report_properties)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run GSM8K evaluation')
    parser.add_argument('--llm_type', default='lambda', 
                       choices=['openai', 'ollama', 'anthropic', 'gemini', 'azure', 'lambda'], 
                       help='LLM type')
    parser.add_argument('--api_key_file', default='../Api_keys/lambda_api.txt', 
                       help='API key file')
    parser.add_argument('--model', default='llama3.1-70b-instruct-berkeley', 
                       help='Model name')
    parser.add_argument('--prompt_file', default='../prompts/CoT_raw.json', 
                       help='Prompt template file')
    parser.add_argument('--temperature', type=float, default=0.1, 
                       help='Temperature for LLM')
    parser.add_argument('--start_index', type=int, default=0, 
                       help='Starting index for processing')
    # Add new dataset argument
    parser.add_argument('--dataset', default='CoT_final', 
                       help='Dataset name (default: GSM8K)')
    args = parser.parse_args()
    llm_config = get_llm_config(args)
    process_questions(llm_config, args.start_index)
