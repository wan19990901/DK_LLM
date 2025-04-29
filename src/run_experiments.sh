#!/bin/bash

# Set default values
DATA="data/final_df_sample.csv"
PROMPT="prompts/CoT_raw.json"
OUTPUT_DIR="results"
TEMPERATURE=0.7

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# List of models to test
MODELS=("gpt-4" "gpt-3.5-turbo" "claude-3-opus" "claude-3-sonnet" "gemini-pro" "grok-1")

# Run experiments for each model
for MODEL in "${MODELS[@]}"; do
    echo "Running experiment with $MODEL..."
    OUTPUT_FILE="$OUTPUT_DIR/${MODEL}_results.csv"
    
    python src/LLM_agent.py \
        --data $DATA \
        --prompt $PROMPT \
        --model $MODEL \
        --output $OUTPUT_FILE \
        --temperature $TEMPERATURE
    
    echo "Results saved to $OUTPUT_FILE"
done

echo "All experiments completed!"