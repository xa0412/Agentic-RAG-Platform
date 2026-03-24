import os
import google.generativeai as genai
import vertexai
from vertexai.language_models import TextEmbeddingModel
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

vertexai.init(
    project=os.getenv("GCP_PROJECT_ID"),
    location=os.getenv("GCP_REGION", "us-central1"),
)


class VertexLLM:
    def __init__(self):
        self.model = genai.GenerativeModel(
            "gemini-2.5-flash-lite",
            generation_config=genai.GenerationConfig(temperature=0),
        )

    class _Response:
        def __init__(self, text: str):
            self.content = text

    def invoke(self, messages: list) -> "_Response":
        prompt = "\n".join(
            f"{msg.__class__.__name__}: {msg.content}"
            for msg in messages
        )
        response = self.model.generate_content(prompt)
        return self._Response(response.text)


class VertexEmbeddings:
    def __init__(self):
        self.model = TextEmbeddingModel.from_pretrained("text-embedding-004")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.get_embeddings(texts, output_dimensionality=256)
        return [e.values for e in embeddings]

    def embed_query(self, text: str) -> list[float]:
        embeddings = self.model.get_embeddings([text], output_dimensionality=256)
        return embeddings[0].values
