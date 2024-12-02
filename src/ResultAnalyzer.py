import sys
import pandas as pd
from pathlib import Path

from Constants import PROJECT_BASE_PATH
from QuestionType import QuestionType
from utils import parse_correct_answer

RESULT_PATH = PROJECT_BASE_PATH / "data" / "results"

def find_file_in_results(file_name, results_path):

    for file in results_path.rglob(file_name):
        return file
    return None

def analyze_correct_answer_in_question():
    question_path = PROJECT_BASE_PATH / "data"
    undefined_answers = []
    for csv_file in question_path.glob("*.csv"):
        binary_answer_index = []
        with open(csv_file, "r") as file:
            df = pd.read_csv(file)
            if "Correct Answer" not in df.columns:
                print(f"'Correct Answer' column not found in {csv_file}")
                continue
            
            for answer in df["Correct Answer"]:
                if parse_correct_answer(answer) == QuestionType.UNDEFINED:
                    undefined_answers.append(answer)
                if parse_correct_answer(answer) == QuestionType.BINARY_CHOICE:
                    binary_answer_index.append(answer)

        print(f"{csv_file} has {len(binary_answer_index)} binary questions.")
    
    # print(undefined_answers)


def main():

    analyze_correct_answer_in_question()

    if len(sys.argv) < 2:
        print("Usage: python3 ResultAnalyzer.py <file1> <file2> ...")
        return

    file_names = sys.argv[1:]

    # Check each file
    for file_name in file_names:
        found_file = find_file_in_results(file_name, RESULT_PATH)
        if found_file:
            print(f"Found: {found_file}")
        else:
            print(f"No file found for: {file_name}")

if __name__ == "__main__":
    main()