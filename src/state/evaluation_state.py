from pydantic import BaseModel, Field


class EvaluationState(BaseModel):
    relevance: str = Field(
        description="Are the documents relevant to the question? 'yes' or 'no'")
