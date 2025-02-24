!/bin/bash

# Run OpenAI GPT-4 Mini experiments
echo "Running OpenAI GPT-4 Mini experiments..."
python main.py --llm_type openai \
               --api_key_file ../Api_keys/openai_api.txt \
               --model gpt-4o-mini \
               --prompt_file ../prompts/base.json \
               --temperature 0

python main.py --llm_type openai \
               --api_key_file ../Api_keys/openai_api.txt \
               --model gpt-4o-mini \
               --temperature 0

# Run Azure GPT-4 Turbo experiments
echo "Running Azure GPT-4 Turbo experiments..."
python main.py --llm_type azure \
               --api_key_file ../Api_keys/azure_api.txt \
               --model gpt-35-turbo \
               --temperature 0.8

python main.py --llm_type azure \
               --api_key_file ../Api_keys/azure_api.txt \
               --model gpt-4-turbo \
               --prompt_file ../prompts/CoT.json \
               --temperature 0

# Run Anthropic Claude experiments
echo "Running Anthropic Claude experiments..."
python main.py --llm_type anthropic \
               --api_key_file ../Api_keys/claude_api.txt \
               --model claude-3-5-haiku-20241022 \
               --prompt_file ../prompts/base.json \
               --temperature 0

python main.py --llm_type anthropic \
               --api_key_file ../Api_keys/claude_api.txt \
               --model claude-3-5-haiku-20241022 \
               --temperature 0.8

# Run Gemini experiments
echo "Running Google Gemini experiments..."
python main.py --llm_type gemini \
               --api_key_file ../Api_keys/gemini_api.txt \
               --model gemini-1.5-flash \
               --prompt_file ../prompts/base.json \
               --temperature 0

python main.py --llm_type gemini \
               --api_key_file ../Api_keys/gemini_api.txt \
               --model gemini-1.5-flash \
               --prompt_file ../prompts/CoT.json \
               --temperature 0

# Run Lambda Llama experiments
echo "Running Lambda Llama experiments..."
python main.py --llm_type lambda \
               --api_key_file ../Api_keys/lambda_api.txt \
               --model llama3.1-70b-instruct-berkeley \
               --prompt_file ../prompts/base.json \
               --temperature 0

python main.py --llm_type lambda \
               --api_key_file ../Api_keys/lambda_api.txt \
               --model llama3.1-70b-instruct-berkeley \
               --prompt_file ../prompts/CoT.json \
               --temperature 0

echo "All experiments completed!"