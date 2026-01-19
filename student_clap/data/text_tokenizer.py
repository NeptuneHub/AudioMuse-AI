from transformers import AutoTokenizer

# Use the same tokenizer as CLAP (usually RoBERTa-base or similar)
TOKENIZER_NAME = "roberta-base"

def get_tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER_NAME)
