from pydantic import BaseModel, Field
from typing import List
from gpt4all import Embed4All

class InputRequest(BaseModel):
    input: str = Field(..., description="The input text")

class Embedding(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int = 0

class OpenAIEmbeddingOutput(BaseModel):
    object: str = "list"
    data: List[Embedding]
    model: str = "gpt4all"

    class Config:
        schema_extra = {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "embedding": [
                        0.0023064255,
                        -0.009327292,
                        -0.0028842222,
                    ],
                    "index": 0,
                }
            ],
            "model": "gpt4all",
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
        }


def get_embedding(data: InputRequest, embedder) -> OpenAIEmbeddingOutput:
    # Perform the embedding calculation here
    embedding = calculate_embedding(data.input, embedder)
    return OpenAIEmbeddingOutput(data=[Embedding(embedding=embedding)])


def calculate_embedding(text, embedder) -> List[float]:
    return embedder.embed(text)
