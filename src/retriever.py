from transformers import DPRRetriever

class DocumentRetriever:
    def __init__(self, model_path):
        self.retriever = DPRRetriever.from_pretrained(model_path)

    def retrieve(self, query, documents):
        return self.retriever(query=query, documents=documents)
