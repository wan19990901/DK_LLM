import re

OPERATORS = {'+', '-', '*', '/', '='}

def parse_answer(answer):
    # Step 1: Check for operators and extract the string after "=" or "= "
    if any(op in answer for op in OPERATORS):
        matches = list(re.finditer(r'=(?:\s*)([^\s]*)', answer))
        
        if matches:
            # Get the last match
            last_match = matches[-1]
            answer = last_match.group(1)  # Extract the component after the last "=" or "= "
        else:
            raise ValueError(f"Answer is operation, cannot find result. '{answer}'")

    # Step 2: Locate the first occurrence of a digit and extract the component
    match = re.search(r'\d', answer)
    if match:
        start_index = match.start()
        match = re.search(r'\S*' + re.escape(answer[start_index]) + r'\S*', answer)
        if match:
            answer = match.group(0)
    else:
        raise ValueError(f"No digit found in answer. '{answer}'")

    # Initialize multiplier
    multiplier = 1
    answer = answer.strip()

    # Step 4: Remove prefix {"$", "-"}
    if answer.startswith('$'):
        answer = answer[1:]
    if answer.startswith('-'):
        answer = answer[1:]
        multiplier *= -1

    # Step 5: Remove suffix {"g", "%"}
    if answer.endswith('%'):
        answer = answer[:-1]
    if answer.endswith('g'):
        answer = answer[:-1]

    # Step 6: Remove potential "," in between digits
    answer = answer.replace(',', '')

    # Step 7: Cast to double, apply multiplier, and round to 2 decimal places
    try:
        parsed_value = float(answer) * multiplier
        return round(parsed_value, 2)
    except ValueError:
        raise ValueError(f"Unable to parse '{answer}' into a float.")
    
def compare_answer(answer, correct_answer):
    try:
        parsed_correct_answer = round(float(correct_answer), 2)
        return answer == parsed_correct_answer
    except ValueError:
        raise ValueError(f"Unable to parse '{correct_answer}' into a float.")