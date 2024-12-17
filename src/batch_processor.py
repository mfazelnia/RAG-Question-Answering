import os
import pickle
from src.retriever import DocumentRetriever
from src.generator import AnswerGenerator
from src.embedder import TextEmbedder

class BatchProcessor:
    def __init__(self):
        self.retriever = DocumentRetriever(model_path="models/retriever_model")
        self.generator = AnswerGenerator(model_path="models/generator_model")
        self.embedder = TextEmbedder()

    def process_questions(self, questions, textbook_path="data/physics_textbook.txt"):
        # Load textbook content
        with open(textbook_path, "r") as file:
            textbook = file.readlines()
        
        # Embed textbook content if not already done
        embeddings_file = "embeddings/embedded_texts.pkl"
        if not os.path.exists(embeddings_file):
            self.embedder.embed_texts(textbook)

        with open(embeddings_file, "rb") as f:
            embedded_texts = pickle.load(f)

        answers = []
        for question in questions:
            # Retrieve context
            docs = [{'text': line} for line in textbook]
            context = self.retriever.retrieve(question, docs)
            # Generate answer
            answer = self.generator.generate(question, context[0]['text'])
            answers.append((question, answer))
        return answers

# Example usage (not part of the module itself):
if __name__ == "__main__":
    questions = ["What is Newton's first law of motion?", "Explain gravitational force."]
    processor = BatchProcessor()
    results = processor.process_questions(questions)
    for q, a in results:
        print(f"Question: {q}\nAnswer: {a}\n")
