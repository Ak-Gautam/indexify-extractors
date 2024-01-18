#code successfully gives the embeddings of input text

from transformers import LEDModel, LEDTokenizer
import torch

class LEDBaseExtractor:
    def __init__(self, model_name="allenai/led-base-16384"):
        # Default model name is "allenai/led-base-16384"
        self.model = LEDModel.from_pretrained(model_name)
        self.tokenizer = LEDTokenizer.from_pretrained(model_name)

    def extract_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

if __name__ == "__main__":
    # Example usage
    extractor = LEDBaseExtractor()
    text = "hey akshay!"
    embedding = extractor.extract_embedding(text)
    print("Embedding:", embedding)
