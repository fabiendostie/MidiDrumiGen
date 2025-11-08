"""Drum Pattern Transformer model implementation."""

import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel
from typing import Optional


class DrumPatternTransformer(nn.Module):
    """
    Autoregressive Transformer for drum pattern generation.
    
    Architecture inspired by GPT-2 with additional style conditioning.
    Trained on tokenized MIDI sequences with style labels.
    """
    
    def __init__(
        self,
        vocab_size: int,
        n_styles: int,
        n_positions: int = 2048,
        n_embd: int = 768,
        n_layer: int = 12,
        n_head: int = 12,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Base Transformer config
        self.config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=dropout,
            embd_pdrop=dropout,
            attn_pdrop=dropout,
        )
        
        # GPT-2 base model
        self.transformer = GPT2LMHeadModel(self.config)
        
        # Style conditioning
        self.style_embeddings = nn.Embedding(n_styles, 128)
        self.style_projection = nn.Linear(128, n_embd)
        self.style_dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights following GPT-2 paper."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        style_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs (batch_size, seq_len)
            style_ids: Style IDs (batch_size,)
            attention_mask: Attention mask (batch_size, seq_len)
            labels: Target tokens for training (batch_size, seq_len)
        
        Returns:
            ModelOutput with loss (if labels provided) and logits
        """
        batch_size, seq_len = input_ids.shape
        
        # Get token embeddings
        inputs_embeds = self.transformer.transformer.wte(input_ids)
        
        # Get style embeddings and project to hidden dim
        style_emb = self.style_embeddings(style_ids)  # (batch, 128)
        style_emb = self.style_projection(style_emb)  # (batch, 768)
        style_emb = self.style_dropout(style_emb)
        
        # Add style to all positions
        inputs_embeds = inputs_embeds + style_emb.unsqueeze(1)
        
        # Forward through Transformer
        outputs = self.transformer(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        
        return outputs
    
    @torch.no_grad()
    def generate(
        self,
        style_id: int,
        max_length: int = 512,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        device: str = "cuda"
    ) -> torch.Tensor:
        """
        Generate drum pattern autoregressively.
        
        Args:
            style_id: Producer style ID
            max_length: Maximum sequence length
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            device: Device to generate on
        
        Returns:
            Generated token IDs
        """
        self.eval()
        
        # Start with BOS token
        BOS_TOKEN_ID = 1
        input_ids = torch.tensor([[BOS_TOKEN_ID]], device=device)
        style_ids = torch.tensor([style_id], device=device)
        
        for _ in range(max_length):
            # Forward pass
            outputs = self.forward(input_ids, style_ids)
            logits = outputs.logits[:, -1, :]  # (batch, vocab_size)
            
            # Apply temperature
            logits = logits / temperature
            
            # Top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = float('-inf')
            
            # Sample from distribution
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Check for EOS token
            EOS_TOKEN_ID = 2
            if next_token.item() == EOS_TOKEN_ID:
                break
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids.squeeze(0)

