from typing import List, Dict, Any

AFFIRMATIVE_QUESTION = "[y/yes to confirm]"
AFFIRMATIVE_RESPONSES = {'y', 'yes'}
OVERRIDE_PREFIX = "Override:"

def override_errors_in_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    
    total_count = len(results)
    error_index = []

    for index in range(total_count):
        if "answer" not in results[index]:
            print("Result does not have answer, canceling errors check.")
            return results
        if results[index]["answer"].startswith(OVERRIDE_PREFIX):
            error_index.append(index)

    print(f"Ambiguous answers count: {len(error_index)}/{total_count}, Start manually override? {AFFIRMATIVE_QUESTION}")

    if input().strip().lower() not in AFFIRMATIVE_RESPONSES:
        return results
    
    print("Please enter override answer, hit enter to skip.")

    for index in range(len(error_index)):
        answer = results[error_index[index]]["answer"]
        print(f"({index+1}/{len(error_index)}) {answer}")
        print("Override with:")
        overrided_answer = input()
        answer = overrided_answer if overrided_answer != "" else answer
        results[error_index[index]]["answer"] = answer

    return results