
# Indexify Extractors

This repository contains a collection of text embedding extractors for use with the Indexify platform. The extractors are implemented in Python and leverage popular natural language processing models.

## LED Extractor

### Model: led-base-16384

The LED (Longformer-Encoder-Decoder) model is a pre-trained transformer-based language model developed by Allen Institute for Artificial Intelligence (AI2). It is designed for efficient and effective text encoding and decoding. The "led-base-16384" variant uses a large embedding dimension of 16384.

#### Usage

To use the LED extractor in your project, follow these steps:

1. Install the required dependencies:

   ```bash
   pip install transformers torch

2.Create an instance of the LEDBaseExtractor class and use it to extract embeddings:

from led_base import LEDBaseExtractor

# Initialize the LED extractor
extractor = LEDBaseExtractor()

# Example text for embedding extraction
text = "This is a sample text."

# Extract embedding
embedding = extractor.extract_embeddings([text])

# Print the embedding
print(embedding)


