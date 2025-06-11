import torch
import torch.nn as nn
import tiktoken
from gpt2 import GPTModel

# Configuration for GPT-2 124M model
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

def encode_text(tokenizer, text):
    """
    Encode input text into token IDs.

    Args:
        tokenizer: Tokenizer from tiktoken.
        text (str): Input prompt.

    Returns:
        torch.Tensor: Shape (1, T) where T is input length.
    """
    encoded = tokenizer.encode(text)
    return torch.tensor(encoded).unsqueeze(0)  # (1, T)

def decode_tokens(tokenizer, tokens):
    """
    Decode token IDs back into text.

    Args:
        tokenizer: Tokenizer from tiktoken.
        tokens (torch.Tensor): Token IDs, shape (1, T).

    Returns:
        str: Decoded string.
    """
    return tokenizer.decode(tokens.squeeze(0).tolist())

def autoregressive_generate(model, input_tokens, num_tokens, context_size):
    """
    Generate a sequence of tokens autoregressively using the GPT model.

    Args:
        model (nn.Module): GPT model.
        input_tokens (torch.Tensor): Initial input tokens, shape (1, T_init).
        num_tokens (int): Number of tokens to generate.
        context_size (int): Max input length the model can attend to.

    Returns:
        torch.Tensor: Extended token sequence, shape (1, T_init + num_tokens).
    """
    for _ in range(num_tokens):
        # Slice to fit model's context window
        input_cond = input_tokens[:, -context_size:]                    # (1, T_cond)

        with torch.no_grad():
            logits = model(input_cond)                                  # (1, T_cond, vocab_size)

        logits = logits[:, -1, :]                                       # (1, vocab_size)
        probs = torch.softmax(logits, dim=-1)                           # (1, vocab_size)
        next_token = torch.argmax(probs, dim=-1, keepdim=True)          # (1, 1)

        # Append new token to sequence
        input_tokens = torch.cat((input_tokens, next_token), dim=1)     # (1, T+1)

    return input_tokens

def run_inference(prompt):
    """
    Perform full model inference: tokenization, generation, decoding.

    Args:
        prompt (str): Input string prompt.

    Returns:
        str: Generated output text.
    """
    tokenizer = tiktoken.get_encoding("gpt2")
    model = GPTModel(GPT_CONFIG_124M)

    input_tokens = encode_text(tokenizer, prompt)                       # (1, T)
    output_tokens = autoregressive_generate(
        model=model,
        input_tokens=input_tokens,
        num_tokens=5,
        context_size=GPT_CONFIG_124M["context_length"]
    )
    return decode_tokens(tokenizer, output_tokens)

if __name__ == '__main__':
    input_prompt = "Hello I am"
    generated_output = run_inference(input_prompt)
    print(generated_output)
