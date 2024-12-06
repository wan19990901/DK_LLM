from pydantic import BaseModel, Field, model_validator


class Base_Parser(BaseModel):
    Answer: str = Field(description="Just the answer only to the provided question without explanation")

    # # You can add custom validation logic easily with Pydantic.
    # @model_validator(mode="before")
    # @classmethod
    # def question_ends_with_question_mark(cls, values: dict) -> dict:
    #     setup = values.get("setup")
    #     if setup and setup[-1] != "?":
    #         raise ValueError("Badly formed question!")
    #     return values


class Confidence_Parser(Base_Parser):
    Confidence: str = Field(description="Confidence for final answer; it should be a floating number from 0 to 1 where 0 to 0.5 means less confident and 0.5 to 1 means more confident")

class CoT_Parser(Base_Parser):
    Reasoning: str = Field(description="Provide your step-by-step reasoning here")
