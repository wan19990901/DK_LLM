from langchain_openai import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.output_parsers import StructuredOutputParser
import json
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_xai import ChatXAI
from langchain_openai import AzureChatOpenAI
import re
import os
from typing import List, Optional, Dict, Any, Tuple, Set
from langchain_community.chat_models import ChatOllama
from Parsers import *
from utils import *
import argparse
import pandas as pd
from dotenv import load_dotenv
import openai
from anthropic import Anthropic
from google.generativeai import GenerativeModel
import groq


class LLM_agent:
    def __init__(self,api_key = None, llm_type = 'openai',model="gpt-4o-mini",temperature=0.3,base_url='https://openrouter.ai/api/v1'):
        self.api_key = api_key
        self.llm_type = llm_type
        self.parser = None
        self.num_of_llm_output = None
        if llm_type == 'openai':
            os.environ["OPENAI_API_KEY"] = self.api_key
            self.llm = ChatOpenAI(model_name=model, temperature=temperature)
        elif llm_type == 'azure':
            os.environ["AZURE_OPENAI_API_KEY"] = self.api_key
            os.environ["AZURE_OPENAI_ENDPOINT"] = "https://rtp2-shared.openai.azure.com/"
            os.environ["AZURE_OPENAI_API_VERSION"] = "2024-10-21"
            os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"] = model
            self.llm = AzureChatOpenAI(
                openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
                azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
            )
        elif llm_type == 'anthropic':
            os.environ["ANTHROPIC_API_KEY"] = self.api_key
            self.llm = ChatAnthropic(model=model, temperature=temperature)
        elif llm_type == 'gemini':
            os.environ["GOOGLE_API_KEY"] = self.api_key
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = '/home/guangya/.config/gcloud/application_default_credentials.json'
            self.llm = ChatGoogleGenerativeAI(model=model, temperature=temperature, gemini_api_key=self.api_key)
        elif self.llm_type == 'xai':
            self.llm = ChatXAI(model=model, temperature=temperature, xai_api_key=self.api_key)
        elif self.llm_type == 'Other':
            self.llm = ChatOpenAI(
                api_key=self.api_key,
                base_url=base_url,  # Add this to your configuration
                model= model)
        elif llm_type == 'ollama':
            self.llm = ChatOllama(model=model, temperature=temperature)
        else:
            raise ValueError(f"Unsupported LLM type: {llm_type}")

    def invoke(self, arg_dict: Dict[str, Any], max_attempts: int = 3) -> Dict[str, Any]:
        """
        Invoke the LLM with the provided arguments.
        Returns the response and number of attempts made.
        """
        if not self.chat_prompt:
            raise ValueError("Prompt template not set. Call setup_prompt first.")

        missing_vars = set(self.chat_prompt.input_variables) - set(arg_dict.keys())
        if missing_vars:
            raise ValueError(f"Missing required input variables: {missing_vars}")

        chain = self.chat_prompt | self.llm
        output = chain.invoke(arg_dict)
        return output.content

    def setup_prompt(self, prompt_json_path: str, parser_obj: BaseModel) -> None:
        """Set up the prompt template and parser"""
        # Load and process the prompt template
        with open(prompt_json_path) as f:
            prompt_data = json.load(f)
            messages = []
            for key, val in prompt_data.items():
                if key == 'system' and self.llm_type != 'ollama':
                    val += '\n{format_instructions}'
                messages.append((key, val))
        self.chat_prompt = ChatPromptTemplate(messages)

    def get_prompt(self):
        return self.chat_prompt
    def get_parser(self):
        return self.parser
    
    def generate_response(self, prompt: str, temperature: float = 0.7) -> str:
        """Generate a response using the specified model."""
        if self.llm_type.startswith('gpt'):
            response = self.llm.invoke({"prompt": prompt})
            return response
        elif self.llm_type.startswith('claude'):
            response = self.llm.invoke({"prompt": prompt})
            return response
        elif self.llm_type.startswith('gemini'):
            response = self.llm.invoke({"prompt": prompt})
            return response
        elif self.llm_type.startswith('grok'):
            response = self.llm.invoke({"prompt": prompt})
            return response
        else:
            raise ValueError(f"Unsupported model: {self.llm_type}")

    def batch_process(self, prompts: List[str], temperature: float = 0.7) -> List[str]:
        """Process multiple prompts in batch."""
        return [self.generate_response(prompt, temperature) for prompt in prompts]

def load_api_keys() -> Dict[str, str]:
    """Load multiple LLM API keys from .env file."""
    load_dotenv()
    required_keys = [
        "OPENAI_API_KEY",
        "OPENAI_API_KEY_rise",
        "OPENAI_API_KEY_yuqi",
        "OPENAI_API_KEY_deepinfra",
        "OPENAI_API_KEY_OR",
        "AZURE_OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_GEMINI_KEY",
        "Grok_API_KEY"
    ]
    api_keys = {key: os.getenv(key) for key in required_keys}
    missing_keys = [key for key, value in api_keys.items() if not value]
    if missing_keys:
        raise ValueError(f"Missing required API keys: {', '.join(missing_keys)}")
    return api_keys

def load_prompt_template(template_path: str) -> Dict:
    """Load a prompt template from a JSON file."""
    with open(template_path, 'r') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description='Generate CoT and confidence responses using LLMs.')
    parser.add_argument('--data', type=str, default='data/final_df_sample.csv', help='Path to the input CSV file')
    parser.add_argument('--prompt', type=str, default='prompts/CoT_raw.json', help='Path to the prompt template JSON file')
    parser.add_argument('--model', type=str, default='gpt-4', help='LLM model to use')
    parser.add_argument('--output', type=str, default='results.csv', help='Path to save the output CSV file')
    parser.add_argument('--temperature', type=float, default=0.7, help='Temperature for response generation')
    args = parser.parse_args()

    # Load API keys
    api_keys = load_api_keys()
    
    # Select the appropriate API key based on the model
    if args.model.startswith('gpt'):
        api_key = api_keys['OPENAI_API_KEY']
    elif args.model.startswith('claude'):
        api_key = api_keys['ANTHROPIC_API_KEY']
    elif args.model.startswith('gemini'):
        api_key = api_keys['GOOGLE_GEMINI_KEY']
    elif args.model.startswith('grok'):
        api_key = api_keys['Grok_API_KEY']
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    # Initialize LLM agent
    agent = LLM_agent(api_key=api_key, llm_type=args.model)

    # Load data
    df = pd.read_csv(args.data)

    # Load prompt template
    prompt_template = load_prompt_template(args.prompt)

    # Generate responses
    results = []
    for idx, row in df.iterrows():
        question = row['Question']
        prompt = prompt_template['template'].format(question=question)
        response = agent.generate_response(prompt, args.temperature)
        results.append({
            'Question': question,
            'Response': response,
            'Model': args.model,
            'Temperature': args.temperature
        })

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output, index=False)
    print(f"Results saved to {args.output}")

if __name__ == '__main__':
    main()