import re

def parse_answer(answer):

    # Step 1: Match the first occurrence of "yes" or "no"
    match = re.search(r'\b(yes|no)\b', answer, re.IGNORECASE)
    if match:
        # Step 2: Return the matched result in lowercase
        return match.group(1).lower()
    else:
        # Raise error if no match is found
        raise ValueError(f"Cannot find binary answer ('yes' or 'no') in the input: '{answer}'")
    
def compare_answer(answer, correct_answer):
    
    answer = answer.strip().lower()
    correct_answer = correct_answer.strip().lower()

    return answer == correct_answer