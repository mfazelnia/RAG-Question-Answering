from transformers import BartForConditionalGeneration, BartTokenizer

class AnswerGenerator:
    def __init__(self, model_path):
        self.model = BartForConditionalGeneration.from_pretrained(model_path)
        self.tokenizer = BartTokenizer.from_pretrained(model_path)

    def generate(self, query, context):
        inputs = self.tokenizer(f"question: {query} context: {context}", return_tensors="pt")
        outputs = self.model.generate(**inputs)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
