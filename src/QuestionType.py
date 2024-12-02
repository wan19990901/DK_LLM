from enum import Enum

class QuestionType(Enum):
    MULTIPLE_CHOICE = 1
    BINARY_CHOICE = 2
    GAP_FILLING_NUMBER = 3
    UNDEFINED = 4