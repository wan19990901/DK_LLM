import re

def parse_answer(answer):
    # Step 1: Strip the input string
    answer = answer.strip()

    # Define patterns for matching
    patterns = [r'^([A-I])(?=\s|$)', r'^([A-I])\)', r'^\(([A-I])\)']
    body_patterns = [r'([A-I])(?=\s|$)', r'([A-I])\)', r'\(([A-I])\)']

    def match_and_validate(text, patterns, check_case=False):
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                choice = match.group(1)
                # Validate "A" for uppercase if required
                if check_case and choice.lower() == 'a' and choice != 'A':
                    continue
                return choice
        return None

    # Step 2: Match at the beginning of the string
    choice = match_and_validate(answer, patterns, check_case=True)
    if choice:
        return choice

    # Step 3: Match in the body of the string
    choice = match_and_validate(answer, body_patterns, check_case=True)
    if choice:
        return choice

    # Step 4: Return the original answer if no match is found
    return answer

def compare_answer(answer, correct_answer, question_text):
    # Step 1: Normalize and directly compare if answer is a single letter A-I
    answer = answer.strip()
    if len(answer) == 1 and answer.lower() in "abcdefghi":
        return answer.lower() == correct_answer.strip().lower()

    # Step 2: Extract correct answer content from question_text
    correct_answer = correct_answer.strip().upper()
    patterns = [
        rf'\n?{correct_answer}(?=\s|$)',          # Match "B" with optional \n in front
        rf'\n?{correct_answer}\)',               # Match "B)" with optional \n in front
        rf'\n?\({correct_answer}\)'              # Match "(B)" with optional \n in front
    ]
    content_match = None

    for pattern in patterns:
        match = re.search(pattern, question_text, re.IGNORECASE)
        if match:
            matched_text = match.group(0)
            if matched_text.strip().lower() == "a":  # Avoid lowercase "a" as a false positive
                continue  

            # Extract content after the match
            start_idx = match.end()
            # Match the next stopping point (e.g., ".", next pattern, or end of string)
            next_match = re.search(
                r'(?<!\d)\.(?!\d)|\n?[A-I]\)|\n?\([A-I]\)',  # Skip "." if surrounded by digits
                question_text[start_idx:], re.IGNORECASE
            )
            end_idx = start_idx + (next_match.start() if next_match else len(question_text[start_idx:]))

            correct_content = question_text[start_idx:end_idx].strip()
            content_match = correct_content
            break

    # Step 3: If no correct content was extracted, return False
    if not content_match:
        return False

    # Step 4: Check if correct_content exists in the answer, ignoring cases
    if content_match.lower() in answer.lower():
        return True

    return False