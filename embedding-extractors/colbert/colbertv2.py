from transformers import AutoTokenizer, AutoModel
from typing import List
from indexify_extractor_sdk import (
    ExtractorSchema,
    EmbeddingSchema,
    Content,
)
from indexify_extractor_sdk.base_embedding import (
    BaseEmbeddingExtractor,
    EmbeddingInputParams,
)
import torch

class ColBERTv2Base(BaseEmbeddingExtractor):
    def __init__(self):
        self.max_context_length = 512  # Set the max_context_length attribute explicitly
        super(ColBERTv2Base, self).__init__(max_context_length=self.max_context_length)
        self._model = AutoModel.from_pretrained('colbert-ir/colbertv2.0', trust_remote_code=True)
        self._tokenizer = AutoTokenizer.from_pretrained('colbert-ir/colbertv2.0')

    def extract_embeddings(self, texts: List[str]) -> List[List[float]]:
        # Tokenize the texts and convert to PyTorch tensors
        encoded_input = self._tokenizer(texts, padding=True, truncation=True, max_length=self.max_context_length, return_tensors='pt')
        # Process tokens through the model
        with torch.no_grad():  # Disable gradient calculation for inference
            model_output = self._model(**encoded_input)
        # Extract the embeddings from the last hidden state
        embeddings = model_output.last_hidden_state[:, 0, :].detach().cpu().numpy().tolist()
        return embeddings

    @classmethod
    def schemas(cls) -> ExtractorSchema:
        return ExtractorSchema(
            features={
                "embedding": EmbeddingSchema(distance_metric="cosine", dim=768)
            },
        )

if __name__ == "__main__":
    extractor = ColBERTv2Base()
    print(extractor.schemas())
    # EmbeddingInputParams() and Content.from_text() defined in the SDK
    embedding_input_params = EmbeddingInputParams()
    contents = [Content.from_text(text="Hello World")]
    print(extractor.extract(contents, embedding_input_params))
