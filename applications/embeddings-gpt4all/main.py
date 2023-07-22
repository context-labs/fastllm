from modal import Stub, web_endpoint, Image, method
from src import InputRequest, get_embedding, OpenAIEmbeddingOutput
from gpt4all import Embed4All
from pydantic import BaseModel

stub = Stub("gpt4all-embeddings")

class Input(BaseModel):
    input: str

image = Image.debian_slim(python_version="3.10").pip_install_from_requirements("./requirements.txt")
@stub.function(image=image)
@web_endpoint(method="POST")
def f(i: Input):
	embedder = Embed4All()
	return get_embedding(InputRequest(input=i.input), embedder)
