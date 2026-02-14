from typing import List


def estimate_token_count(text: str) -> int:
    """
    Estimate token count without loading tokenizer.
    Rough approximation: ~4 characters per token for English.
    """
    return len(text) // 4


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """
    Truncate text to approximately max_tokens.
    """
    estimated_chars = max_tokens * 4
    if len(text) <= estimated_chars:
        return text
    return text[:estimated_chars]


def count_tokens_accurate(text: str, tokenizer) -> int:
    """
    Accurate token count using actual tokenizer.
    """
    return len(tokenizer.encode(text))


def split_into_chunks(text: str, max_tokens: int, tokenizer) -> List[str]:
    """
    Split text into chunks of max_tokens.
    """
    tokens = tokenizer.encode(text)
    chunks = []
    
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
    
    return chunks
