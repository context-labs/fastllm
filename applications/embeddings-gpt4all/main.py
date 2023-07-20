from fastapi import FastAPI
from src import InputRequest, get_embedding, OpenAIEmbeddingOutput
from modal import Image, Stub, asgi_app

app = FastAPI()
stub = Stub()


@app.post("/v1/embeddings", response_model=OpenAIEmbeddingOutput)
def process_embedding(data: InputRequest):
    return get_embedding(data)


@asgi_app()
def fastapi_app():
    return app