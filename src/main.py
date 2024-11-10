import argparse
from LLM_agent import LLM_agent
from Parsers import Base_Parser
import os
import pandas as pd
from tqdm import tqdm
import json
from typing import Dict, List, Any
from datetime import datetime

DATA_DIR = '../data/'

# Experiment Config
DF_NAME = 'GSM8K'
NUM_OF_SAMPLES = 2
NUM_OF_REPEAT = 2

def get_llm_config(args) -> Dict[str, Any]:
    """Get LLM configuration from arguments"""
    return {
        'llm_type': args.llm_type,
        'api_key_link': args.api_key_file,
        'model': args.model,
        'prompt_link': args.prompt_file,
        'parser_template': Base_Parser,
        'temperature': args.temperature,    
    }

def save_json(results: Dict[str, Any], llm_config: Dict[str, Any]) -> None:
    """Save results to a JSON file"""
    storage_dir = f'../data/results/{llm_config["model"]}'
    if not os.path.exists(storage_dir):
        os.makedirs(storage_dir)
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(storage_dir, f'{DF_NAME}_{timestamp}.json')
    
    # Add metadata
    results['metadata'] = {
        'prompt_file': os.path.basename(llm_config['prompt_link']),
        'num_samples': NUM_OF_SAMPLES,
        'llm_type': llm_config['llm_type'],
        'temperature': llm_config['temperature']
    }
    
    with open(file_path, 'w') as f:
        json.dump(results, f, indent=2)

def process_questions(llm_config: Dict[str, Any], start_index: int = 0) -> None:
    """Process questions and collect results"""
    MAX_PARSE_ATTEMPTS = 3  # Maximum number of attempts for parsing errors
    
    with open(llm_config['api_key_link'], 'r') as f:
        api_key = f.read().strip()

    df = pd.read_csv(os.path.join(DATA_DIR, f'{DF_NAME}.csv'))
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

    # Process each question
    for row_idx in tqdm(range(start_index, len(df_subset)), colour='blue', desc='Questions', position=0):
        row = df_subset.iloc[row_idx]
        
        # Initialize agent once per question
        cot_agent = LLM_agent(
            llm_type=llm_config['llm_type'], 
            api_key=api_key, 
            model=llm_config['model'],
            temperature=llm_config['temperature']
        )
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
                        'confidence': response['Confidence']
                    })
                    success = True
                except Exception as e:
                    last_error = str(e)
                    parse_attempts += 1
                    if parse_attempts < MAX_PARSE_ATTEMPTS:
                        print(f"Attempt {parse_attempts} failed: {last_error}, retrying...")
            
            # If we haven't successfully created a result entry, create one with error info
            if not success:
                result_entry = base_result.copy()
                result_entry.update({
                    'repeat_number': repeat,
                    'answer': f"Error after {MAX_PARSE_ATTEMPTS} attempts: {last_error}",
                    'confidence': None
                })
            
            results_dict['results'].append(result_entry)
            
            results_dict['results'].append(result_entry)

        # Save intermediate results after each question
        save_json(results_dict, llm_config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run GSM8K evaluation')
    parser.add_argument('--llm_type', default='openai', 
                       choices=['openai', 'ollama', 'anthropic', 'gemini', 'azure'], 
                       help='LLM type')
    parser.add_argument('--api_key_file', default='../Api_keys/openai_api.txt', 
                       help='API key file')
    parser.add_argument('--model', default='gpt-3.5-turbo-0125', 
                       help='Model name')
    parser.add_argument('--prompt_file', default='../prompts/base.json', 
                       help='Prompt template file')
    parser.add_argument('--temperature', type=float, default=0, 
                       help='Temperature for LLM')
    parser.add_argument('--start_index', type=int, default=0, 
                       help='Starting index for processing')
    
    args = parser.parse_args()
    llm_config = get_llm_config(args)
    process_questions(llm_config, args.start_index)
