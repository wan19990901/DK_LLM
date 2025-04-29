import time
import numpy as np
# Function to load GloVe vectors
import json
import re
from copy import deepcopy
import os
import pandas as pd
from collections import Counter

from Constants import BINARY_CHOICE_SET, MULTIPLE_CHOICE_SET
from QuestionType import QuestionType

def parse_correct_answer(correct_answer):
    correct_answer = str(correct_answer).strip()
    if correct_answer[0].isdigit() or (correct_answer[0] == "-" and len(correct_answer) > 1 and correct_answer[1].isdigit()):
        return QuestionType.GAP_FILLING_NUMBER
    if correct_answer[0].isalpha() and len(correct_answer) == 1 and correct_answer[0].lower() in MULTIPLE_CHOICE_SET:
        return QuestionType.MULTIPLE_CHOICE
    if correct_answer[0].isalpha() and len(correct_answer) > 1  and correct_answer.lower() in BINARY_CHOICE_SET:
        return QuestionType.BINARY_CHOICE
    return QuestionType.UNDEFINED

# This is a little bit hard coded, but it's fine for now; need to change the flexibility of the parser_template
def extract_json(text):
    # Find the first opening { and last closing } bracket
    start = text.find('{')
    end = text.rfind('}')
    
    if start == -1 or end == -1:
        return text
        
    # Extract the JSON string
    json_str = text[start:end + 1]
    
    return json_str
    
def extract_nums(s):
    s = s.replace(",", "")
    nums = re.findall(r"[+-]? *(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", s)
    return_list = []
    for i in range(len(nums)):
        try:
            return_list.append(eval(nums[i].strip().lstrip(" 0")))
        except:
            pass
    return return_list

def find_formula(step):
    assert step.count("<<") == step.count(">>") == 1
    left, right = step.find("<<")+2, step.find(">>")
    return step[left: right]


def extract_answer(completion):
    ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        assert False


def delete_extra_zero(n):
    try:
        n=float(n)
    except:
        print("None {}".format(n))
        return n
    if isinstance(n, int):
        return str(n)
    if isinstance(n, float):
        n = str(n).rstrip('0')  # 删除小数点后多余的0
        n = int(n.rstrip('.')) if n.endswith('.') else float(n)  # 只剩小数点直接转int，否则转回float
        n=str(n)
        return n


def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string


def _remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        # assert len(splits) == 2
        return splits[0]
    else:
        return string


def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def _strip_string(string):
    # linebreaks
    string = string.replace("\n", "")
    # print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    # print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    # print(string)

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    # print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    # print(string)

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string

def calculate_SC_correctness(df):
    def check_majority(answers, correct):
        if not answers:
            return 0
        
        # Convert answers to floats if possible, otherwise keep as strings
        converted_answers = []
        for answer in answers:
            try:
                converted_answers.append(float(answer))
            except ValueError:
                converted_answers.append(answer.strip().lower())
        
        # Convert correct answer to float if possible
        try:
            correct_float = float(correct)
        except ValueError:
            correct_float = correct.strip().lower()
        
        # Count the occurrences of each answer and find the most common one
        most_common = Counter(converted_answers).most_common(1)[0][0]
        
        # Compare the most common answer with the correct answer
        return 1 if most_common == correct_float else 0

    # Apply the helper function across the DataFrame rows
    df['SC_correctness'] = df.apply(lambda row: check_majority(row['CoT answers'], row['correct answer']), axis=1)
    return df

def extract_option_logprobs(logprobs_dict):
    """
    Extracts the log probability of the chosen option token (A-H) following 
    the 'answer' token in a logprobs dictionary.

    Args:
        logprobs_dict (str | dict): A string representation or dictionary 
                                    containing logprobs information. Expected 
                                    structure includes a 'content' list with 
                                    token dictionaries, each having 'token' 
                                    and 'logprob'.

    Returns:
        float: The log probability of the first option token found after 
               'answer', or -1 if not found or an error occurs.
    """
    try:
        # Evaluate if it's a string representation
        if isinstance(logprobs_dict, str):
            try:
                # Basic safety check before eval
                if '{' not in logprobs_dict or '}' not in logprobs_dict:
                     print(f"Warning: eval may fail on non-dict string: {logprobs_dict[:100]}")
                logprobs_dict = eval(logprobs_dict)
            except Exception as e:
                print(f"Error evaluating logprobs string: {str(e)}")
                return -1
                
        if not isinstance(logprobs_dict, dict) or 'content' not in logprobs_dict:
             print(f"Invalid logprobs format: {logprobs_dict}")
             return -1
             
        tokens = logprobs_dict.get('content', []) # Use .get for safety
        if not isinstance(tokens, list):
            print("Invalid 'content' format, not a list.")
            return -1

        final_idx = None
        for i, token_info in enumerate(tokens):
            if not isinstance(token_info, dict):
                # print(f"Skipping invalid token_info (not a dict): {token_info}")
                continue # Skip non-dict items
            # Handle potential variations like ' answer' or ':answer'
            # Use .get with default empty string for safety
            if 'answer' in token_info.get('token', '').strip().lower():
                final_idx = i
                break
                
        if final_idx is None:
            # print("Could not find 'answer' token.")
            return -1
            
        # Look for option token in the next few tokens (increased range slightly)
        for i in range(final_idx + 1, min(final_idx + 6, len(tokens))):
            if i >= len(tokens) or not isinstance(tokens[i], dict):
                 continue # Boundary and type check
            token = tokens[i].get('token', '').strip()
            # Check if token matches pattern '[A-H]'
            if len(token) == 1 and token[0] in 'ABCDEFGH':
                return tokens[i].get('logprob', -1) # Return logprob or -1 if missing
                    
        # print(f"Option token not found near 'answer'.") # Simplified debug message
        return -1
    except Exception as e:
        # Catch broader exceptions during processing
        print(f"General error processing logprobs: {str(e)}")
        return -1

def extract_complete_answer(question_text, correct_answer_letter):
    """
    Extracts the full text of the correct answer option from the question text.
    Handles multiple-choice questions with options potentially separated by 
    newlines or commas, prefixed with letters (A, B, C, etc.). Also handles
    simple Yes/No answers directly.

    Args:
        question_text (str): The full text of the question including options.
        correct_answer_letter (str): The letter (or Yes/No) corresponding to the 
                                     correct answer.

    Returns:
        str: The full text of the correct option, or the original 
             correct_answer_letter if extraction fails or it's Yes/No.
    """
    # Ensure inputs are strings
    question_text = str(question_text) if question_text is not None else ""
    correct_answer_letter = str(correct_answer_letter).strip() if correct_answer_letter is not None else ""
    
    if not correct_answer_letter:
        return "" # Return empty if correct answer is missing
        
    # Handle Yes/No answers directly
    if correct_answer_letter.lower() in ['yes', 'no']:
        # Return with consistent casing maybe?
        return correct_answer_letter.capitalize()
    
    # Pattern to find options like A), A ), A. B), B ), B. etc.
    # Looks for the correct letter (case-insensitive), optional space, delimiter [.)], optional space,
    # captures the text until the next option pattern (letter[.)]) or end of string.
    pattern = re.compile(
        rf"^\s*(?:{re.escape(correct_answer_letter)})\s*[.)]\s*(.*?)(?=\n\s*[A-Z]\s*[.)]|\Z)", 
        re.MULTILINE | re.DOTALL | re.IGNORECASE
    )
    
    match = pattern.search(question_text)
    
    if match:
        # Return the captured group (the option text), stripped
        return match.group(1).strip()
    else:
        # Fallback: try finding based on simple newline splitting if pattern fails
        # Handle both \n and actual newlines
        lines = question_text.replace('\\n', '\n').split('\n')
        for line in lines:
             line_strip = line.strip()
             # Check startswith ignoring case and with flexible delimiters
             if re.match(rf"^\s*{re.escape(correct_answer_letter)}\s*[.)]", line_strip, re.IGNORECASE):
                 # Extract text after the marker (e.g., "A) ", "B. ")
                 option_text = re.sub(rf"^\s*{re.escape(correct_answer_letter)}\s*[.)]\s*", "", line_strip, flags=re.IGNORECASE)
                 return option_text.strip()

        # If no match found after trying patterns and lines, return the original letter
        # print(f"Warning: Could not extract complete answer for '{correct_answer_letter}' from question.")
        return correct_answer_letter # Return original letter as fallback

# Example Usage (can be removed or kept for testing):
# question_example = '''What is the capital of France?
# A) London
# B ) Paris
# C. Berlin
# D) Madrid'''
# print(f"Extracted for B: {extract_complete_answer(question_example, 'B')}")
# print(f"Extracted for C: {extract_complete_answer(question_example, 'C')}")

# question_example_yesno = "Is the sky blue?"
# print(f"Extracted for Yes: {extract_complete_answer(question_example_yesno, 'Yes')}")