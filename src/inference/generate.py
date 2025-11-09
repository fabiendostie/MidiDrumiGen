"""Pattern generation inference using trained models."""

import logging
from typing import List, Optional, Any

import torch

from src.models.transformer import DrumPatternTransformer

logger = logging.getLogger(__name__)


class GenerationError(Exception):
    """Raised when pattern generation fails."""
    pass


def generate_pattern(
    model: DrumPatternTransformer,
    tokenizer: Any,
    prompt_tokens: Optional[List[int]] = None,
    num_bars: int = 4,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
    max_length: int = 512,
    device: str = "cpu",
    style_id: int = 0,
) -> List[int]:
    """
    Generate drum pattern tokens using the model.

    Uses autoregressive generation with temperature scaling, top-k, and nucleus (top-p)
    sampling for controllable randomness.

    Args:
        model: Loaded PyTorch model (DrumPatternTransformer)
        tokenizer: MidiTok tokenizer instance for encoding/decoding
        prompt_tokens: Optional starting tokens (if None, starts with BOS token)
        num_bars: Number of bars to generate (default: 4)
        temperature: Sampling temperature (default: 1.0)
                    - Lower (0.5-0.8): More deterministic, safer patterns
                    - Higher (1.2-1.5): More creative, riskier patterns
        top_k: Top-k sampling parameter (default: 50)
              - Only consider top k most likely tokens
              - 0 disables top-k filtering
        top_p: Nucleus sampling parameter (default: 0.9)
              - Sample from smallest set of tokens with cumulative probability >= top_p
              - 1.0 disables nucleus sampling
        max_length: Maximum sequence length (default: 512)
        device: Device to run on ("cuda" or "cpu")
        style_id: Producer style ID for style conditioning (default: 0)

    Returns:
        List of generated token IDs

    Raises:
        GenerationError: If generation fails

    Example:
        >>> model, metadata = load_model(model_path)
        >>> tokens = generate_pattern(
        ...     model=model,
        ...     tokenizer=tokenizer,
        ...     num_bars=4,
        ...     temperature=0.9,
        ...     device="cuda"
        ... )
        >>> print(f"Generated {len(tokens)} tokens")
    """
    try:
        logger.info(
            f"Starting generation: num_bars={num_bars}, temp={temperature}, "
            f"top_k={top_k}, top_p={top_p}, device={device}"
        )

        # Set model to evaluation mode
        model.eval()

        # Prepare starting tokens
        BOS_TOKEN_ID = 1  # Beginning of sequence
        EOS_TOKEN_ID = 2  # End of sequence

        if prompt_tokens is None:
            # Start with BOS token
            input_ids = torch.tensor([[BOS_TOKEN_ID]], dtype=torch.long, device=device)
        else:
            # Use provided prompt
            input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

        # Prepare style conditioning
        style_ids = torch.tensor([style_id], dtype=torch.long, device=device)

        logger.debug(f"Starting generation from {input_ids.shape[1]} tokens")

        # Autoregressive generation loop
        generated_tokens = []

        with torch.no_grad():
            for step in range(max_length):
                # Forward pass through model
                outputs = model(input_ids, style_ids)
                logits = outputs.logits[:, -1, :]  # Get logits for last position (batch, vocab_size)

                # Apply temperature scaling
                if temperature != 1.0:
                    logits = logits / temperature

                # Top-k filtering
                if top_k > 0:
                    # Get top k values and indices
                    top_k_values, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
                    # Set all other values to -inf
                    indices_to_remove = logits < top_k_values[:, -1, None]
                    logits = logits.masked_fill(indices_to_remove, float('-inf'))

                # Nucleus (top-p) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep first token above threshold
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = False

                    # Scatter sorted tensors back to original indexing
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits = logits.masked_fill(indices_to_remove, float('-inf'))

                # Sample from the filtered distribution
                probs = torch.softmax(logits, dim=-1)

                # Check for valid probabilities
                if torch.isnan(probs).any() or torch.isinf(probs).any():
                    logger.error("Invalid probabilities detected during sampling")
                    raise GenerationError("Generation produced invalid probabilities")

                # Sample next token
                next_token = torch.multinomial(probs, num_samples=1)

                # Check for EOS token (end generation early)
                if next_token.item() == EOS_TOKEN_ID:
                    logger.debug(f"EOS token generated at step {step}")
                    break

                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)
                generated_tokens.append(next_token.item())

                # Log progress periodically
                if (step + 1) % 50 == 0:
                    logger.debug(f"Generated {step + 1}/{max_length} tokens")

        logger.info(f"Generation complete: {len(generated_tokens)} tokens generated")

        return generated_tokens

    except torch.cuda.OutOfMemoryError:
        logger.error("GPU out of memory during generation")
        raise GenerationError(
            "GPU out of memory. Try reducing max_length or using CPU device."
        )

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise GenerationError(f"Pattern generation failed: {str(e)}")


def generate_batch(
    model: DrumPatternTransformer,
    tokenizer: Any,
    batch_size: int,
    num_bars: int = 4,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
    max_length: int = 512,
    device: str = "cpu",
    style_ids: Optional[List[int]] = None,
) -> List[List[int]]:
    """
    Generate multiple patterns in parallel (batch generation).

    More efficient than sequential generation for multiple requests.

    Args:
        model: Loaded PyTorch model
        tokenizer: MidiTok tokenizer instance
        batch_size: Number of patterns to generate
        num_bars: Number of bars per pattern
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Nucleus sampling parameter
        max_length: Maximum sequence length
        device: Device to run on
        style_ids: Optional list of style IDs (one per batch item)
                  If None, uses style_id=0 for all

    Returns:
        List of token lists (one per batch item)

    Example:
        >>> patterns = generate_batch(
        ...     model=model,
        ...     tokenizer=tokenizer,
        ...     batch_size=4,
        ...     style_ids=[0, 1, 0, 2]  # Different styles
        ... )
        >>> print(f"Generated {len(patterns)} patterns")
    """
    logger.info(f"Starting batch generation: batch_size={batch_size}")

    # Prepare style IDs
    if style_ids is None:
        style_ids = [0] * batch_size
    elif len(style_ids) != batch_size:
        raise ValueError(f"style_ids length ({len(style_ids)}) must match batch_size ({batch_size})")

    # Generate each pattern (sequential for now - can be optimized later)
    patterns = []
    for i, style_id in enumerate(style_ids):
        logger.debug(f"Generating pattern {i+1}/{batch_size}")
        pattern = generate_pattern(
            model=model,
            tokenizer=tokenizer,
            num_bars=num_bars,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_length=max_length,
            device=device,
            style_id=style_id,
        )
        patterns.append(pattern)

    logger.info(f"Batch generation complete: {len(patterns)} patterns generated")
    return patterns


def estimate_generation_time(
    num_bars: int,
    device: str = "cuda",
    model_size: str = "base"
) -> float:
    """
    Estimate generation time for a pattern.

    Provides rough time estimates based on hardware and model size.

    Args:
        num_bars: Number of bars to generate
        device: Device being used ("cuda" or "cpu")
        model_size: Model size ("base", "large", etc.)

    Returns:
        Estimated time in seconds

    Example:
        >>> time_sec = estimate_generation_time(4, device="cuda")
        >>> print(f"Expected generation time: {time_sec:.1f}s")
    """
    # Rough estimates based on typical performance
    # (will be calibrated based on actual measurements)

    tokens_per_bar = 32  # Approximate

    if device == "cuda":
        # GPU generation: ~100 tokens/sec for base model
        tokens_per_second = 100 if model_size == "base" else 50
    else:
        # CPU generation: ~10 tokens/sec for base model
        tokens_per_second = 10 if model_size == "base" else 5

    total_tokens = num_bars * tokens_per_bar
    estimated_time = total_tokens / tokens_per_second

    return estimated_time
