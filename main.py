import os
from dotenv import load_dotenv

from openai import OpenAI
from pydantic import BaseModel
import pandas as pd


load_dotenv()

client = OpenAI(
    api_key=os.environ.get("OLLAMA_API_KEY"),
    base_url=os.environ.get("BASE_URL")
)

class CodeReview(BaseModel):
    """Schema for structured code review output."""
    language: str
    summary: str
    issues: list[str]
    severity: str  # e.g., "low", "medium", "high"


response = client.responses.parse(
    model=os.environ.get("LLM_MODEL"),
    instructions="Review the given code. Respond in the structured format provided.",
    input="Review this code: def add(a, b): return a + b",
    text_format=CodeReview,
)

review = response.output_parsed  # This is a CodeReview instance

# print(f"Language : {review.language}")
# print(f"Summary  : {review.summary}")
# print(f"Issues   : {review.issues}")
# print(f"Severity : {review.severity}")

df = pd.DataFrame([review.model_dump()])

print(df)
