# DK_LLM
Github repo for Dunning Kruger Effect quantification of LLM

# LLM Calibration Analysis

This project analyzes the calibration of various LLMs by generating Chain-of-Thought (CoT) and confidence responses, and evaluating their performance.

## Project Structure

```
DK_LLM/
├── src/                    # Source code
│   ├── data_loader.py     # Data loading and preprocessing
│   ├── evaluation.py      # Evaluation metrics and analysis
│   ├── plotting.py        # Visualization and plotting
│   ├── utils.py           # Utility functions
│   ├── main.py            # Main execution script
│   ├── LLM_agent.py       # LLM interaction and response generation
│   └── parsers/           # Response parsing utilities
├── data/                   # Input data and generated results
├── prompts/               # Prompt templates for different LLM configurations
├── calibration_analysis_plots/  # Generated plots and analysis results
└── analysis.ipynb         # Jupyter notebook for interactive analysis
```

## Environment Setup

1. Install the required dependencies:
   ```bash
   conda env create -f enviroments.yml
   conda activate llm_calibration
   ```

2. Create a `.env` file in the project root with your API keys:
   ```
   OPENAI_API_KEY=your_key
   ANTHROPIC_API_KEY=your_key
   GOOGLE_GEMINI_KEY=your_key
   Grok_API_KEY=your_key
   ```

## Usage

### Generating CoT and Confidence Responses

Use the `LLM_agent.py` script to generate responses for a set of questions:

```bash
python src/LLM_agent.py --data data/final_df_sample.csv --prompt prompts/CoT_raw.json --model gpt-4 --output results.csv
```

### Running Experiments

Use the provided shell script to run experiments:

```bash
bash src/run_experiments.sh
```

### Interactive Analysis

Use the Jupyter notebook for interactive analysis:

```bash
jupyter notebook analysis.ipynb
```

## Supported LLM Models

The project supports the following LLM models:
- OpenAI (GPT-3.5, GPT-4)
- Anthropic (Claude)
- Google (Gemini)
- Grok
- Azure OpenAI

## Notes

- Ensure all API keys are correctly set in the `.env` file before running the generation script.
- The project supports batch processing for efficient generation of responses.
- Results are saved in CSV format and can be analyzed using the provided tools.
- Visualization tools are available in the `plotting.py` module.
