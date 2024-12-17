import os
import pickle
from src.retriever import DocumentRetriever
from src.generator import AnswerGenerator
from src.embedder import TextEmbedder

# Initialize models
retriever = DocumentRetriever(model_path="models/retriever_model")
generator = AnswerGenerator(model_path="models/generator_model")
embedder = TextEmbedder()

# Embed textbook content
with open("data/physics_textbook.txt", "r") as file:
    textbook = file.readlines()
    embedder.embed_texts(textbook)

# Load embeddings
with open("embeddings/embedded_texts.pkl", "rb") as f:
    embedded_texts = pickle.load(f)

# Main Loop
while True:
    question = input("Enter your physics question (or 'quit' to exit): ")
    if question.lower() == 'quit':
        break
    # Retrieve relevant text
    docs = [{'text': line} for line in textbook]
    context = retriever.retrieve(question, docs)
    # Generate answer
    answer = generator.generate(question, context[0]['text'])
    print(f"Answer: {answer}\n")
