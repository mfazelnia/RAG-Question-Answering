from openai import OpenAIEmbeddings
import pickle

class TextEmbedder:
    def __init__(self, model_name="text-embedding-ada-002"):
        self.embedder = OpenAIEmbeddings(model=model_name)

    def embed_texts(self, texts):
        embeddings = self.embedder.embed_documents(texts)
        with open("embeddings/embedded_texts.pkl", "wb") as f:
            pickle.dump(embeddings, f)
        return embeddings
