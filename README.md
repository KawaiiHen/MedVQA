# Medical Visual Question Answering (VQA) with BLIP and PathVQA

This project focuses on building a Medical Visual Question Answering (VQA) system using the BLIP model and the PathVQA dataset. The goal is to analyze medical images and answer corresponding questions about the image content. 

## Project Overview

Medical Visual Question Answering (VQA) involves using machine learning models to automatically provide answers to questions about medical images. This project uses the BLIP (Bootstrapped Language-Image Pretraining) model, fine-tuned with the PathVQA dataset, which contains questions and answers related to medical images.

The workflow involves mainly these 5 steps and some keynote :

1. Loading and preprocessing the PathVQA dataset.

2. Tokenizing text (questions and answers) and processing images.
   
3. Extending the BLIP model to handle the medical vocabulary.
   
*As Medical VQA requires domain-specific vocabulary. The project's tokenizer is expanded to include unknown tokens present in the PathVQA dataset, allowing the model to handle medical terminology.*
```python
# Tokenize dataset and add unknown tokens
unknown_tokens = set(token_counter.keys()) - vocab_tokens
text_processor.tokenizer.add_tokens(list(unknown_tokens))
```
  
4. Training the model on the PathVQA dataset.
   
*The model's text embeddings and output layers are extended to handle the newly added tokens*
```python
# Extend the embedding and linear layers
new_embeddings = torch.nn.Embedding(new_vocab_size, current_embeddings.shape[1])
model.text_encoder.embeddings.word_embeddings = new_embeddings
```

5. Evaluating the model and visualizing the predictions.

## Dataset

We use the **PathVQA** dataset, available via Hugging Face. This dataset consists of medical images, each paired with a question and an answer. The dataset is divided into training and validation sets.

```python
from datasets import load_dataset
dataset = load_dataset("flaviagiammarino/path-vqa")
```

## Model Architecture
The project leverages the BLIP (Bootstrapped Language-Image Pretraining) model, which is pre-trained for visual question answering tasks. The architecture (Kindly check our anaylysis of the BLIP model in the FINAL_DL.docx)  includes: 

1. Image Encoder: Extracts visual features from medical images.
2. Text Encoder: Processes the questions and answers.
3. Vocabulary Expansion: Customizes the model's vocabulary to handle specialized medical terms.
