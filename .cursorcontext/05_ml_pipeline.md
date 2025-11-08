# ML Pipeline - Context Document

**Document Type:** Knowledge Context (Machine Learning)  
**Purpose:** Model architecture, training pipeline, inference optimization  
**Use Case:** Load when implementing ML features, training, or debugging generation

---

## Model Architecture

### DrumPatternTransformer Overview

**Base Architecture:** GPT-2 style autoregressive Transformer  
**Purpose:** Generate drum patterns conditioned on producer style

**Key Components:**
1. Token Embeddings (vocab → hidden dim)
2. Style Conditioning (learned embeddings)
3. Positional Encodings (absolute positions)
4. Transformer Encoder (12 layers, 768-dim, 12 heads)
5. Output Head (hidden → vocab logits)

### Implementation

```python
# src/models/transformer.py
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
```

---

## Training Pipeline

### Data Preparation

```python
# src/training/dataset.py
import torch
from torch.utils.data import Dataset
from pathlib import Path
import json

class DrumPatternDataset(Dataset):
    """Dataset for tokenized drum patterns."""
    
    def __init__(
        self,
        tokens_dir: Path,
        styles_path: Path,
        max_length: int = 512,
    ):
        self.tokens_dir = Path(tokens_dir)
        self.max_length = max_length
        
        # Load style mappings
        with open(styles_path) as f:
            self.style_to_id = json.load(f)
        
        # Load all token files
        self.token_files = list(self.tokens_dir.glob("*.pt"))
        
    def __len__(self):
        return len(self.token_files)
    
    def __getitem__(self, idx):
        # Load tokens
        token_path = self.token_files[idx]
        tokens = torch.load(token_path)
        
        # Get style from filename (e.g., "jdilla_001.pt")
        style_name = token_path.stem.split('_')[0]
        style_id = self.style_to_id[style_name]
        
        # Truncate or pad to max_length
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            PAD_TOKEN_ID = 0
            padding = torch.full(
                (self.max_length - len(tokens),),
                PAD_TOKEN_ID,
                dtype=tokens.dtype
            )
            tokens = torch.cat([tokens, padding])
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = (tokens != 0).long()
        
        return {
            'input_ids': tokens[:-1],  # All but last token
            'labels': tokens[1:],      # All but first token
            'style_ids': torch.tensor(style_id, dtype=torch.long),
            'attention_mask': attention_mask[:-1],
        }
```

### Training Script

```python
# src/training/train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
import wandb
from tqdm import tqdm

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 100,
    learning_rate: float = 5e-5,
    device: str = "cuda",
    checkpoint_dir: Path = Path("checkpoints"),
    use_wandb: bool = True,
):
    """
    Train drum pattern generation model.
    
    Uses:
    - AdamW optimizer with weight decay
    - Mixed precision training (AMP)
    - Gradient accumulation
    - Learning rate warmup and decay
    - Gradient clipping
    """
    
    model.to(device)
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.999),
    )
    
    # Learning rate scheduler (warmup + cosine decay)
    from torch.optim.lr_scheduler import OneCycleLR
    scheduler = OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,  # 10% warmup
    )
    
    # Mixed precision training
    scaler = GradScaler()
    
    # Initialize wandb
    if use_wandb:
        wandb.init(project="MidiDrumiGen", config={
            "learning_rate": learning_rate,
            "epochs": num_epochs,
            "batch_size": train_loader.batch_size,
        })
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in progress_bar:
            # Move to device
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            style_ids = batch['style_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass with mixed precision
            with autocast(device_type='cuda'):
                outputs = model(
                    input_ids=input_ids,
                    style_ids=style_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
            
            # Backward pass
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            
            # Track loss
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        val_loss = evaluate_model(model, val_loader, device)
        
        # Logging
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        if use_wandb:
            wandb.log({
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
                "learning_rate": scheduler.get_last_lr()[0],
            })
        
        # Save checkpoint if best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = checkpoint_dir / f"best_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")
        
        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = checkpoint_dir / f"epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)

@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    val_loader: DataLoader,
    device: str = "cuda",
) -> float:
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        style_ids = batch['style_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            style_ids=style_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        total_loss += outputs.loss.item()
    
    return total_loss / len(val_loader)
```

### Training Configuration

```yaml
# configs/base.yaml
model:
  vocab_size: 1000
  n_styles: 50
  n_positions: 2048
  n_embd: 768
  n_layer: 12
  n_head: 12
  dropout: 0.1

training:
  batch_size: 32
  num_epochs: 100
  learning_rate: 5e-5
  weight_decay: 0.01
  gradient_accumulation_steps: 1
  max_grad_norm: 1.0
  warmup_ratio: 0.1

data:
  train_split: 0.9
  val_split: 0.1
  max_length: 512
  shuffle: true

optimization:
  mixed_precision: true
  compile_model: false  # PyTorch 2.0 compile (experimental)

logging:
  use_wandb: true
  log_interval: 10
  eval_interval: 500

checkpointing:
  save_interval: 10
  keep_last_n: 3
```

---

## Inference Optimization

### Model Loading and Caching

```python
# src/inference/model_loader.py
from functools import lru_cache
import torch
from pathlib import Path

@lru_cache(maxsize=10)
def load_model(
    style_name: str,
    checkpoint_dir: Path = Path("models/checkpoints"),
    device: str = "cuda",
) -> DrumPatternTransformer:
    """
    Load model with caching.
    
    Uses LRU cache to keep frequently used models in memory.
    Supports up to 10 different style models simultaneously.
    """
    checkpoint_path = checkpoint_dir / f"{style_name}_v1.pt"
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model not found: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Initialize model
    model = DrumPatternTransformer(
        vocab_size=1000,
        n_styles=50,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model
```

### Batch Generation

```python
def generate_batch(
    model: DrumPatternTransformer,
    style_ids: list[int],
    max_length: int = 512,
    device: str = "cuda",
) -> list[torch.Tensor]:
    """
    Generate multiple patterns in parallel.
    
    More efficient than sequential generation for multiple requests.
    """
    batch_size = len(style_ids)
    
    # Initialize with BOS tokens
    input_ids = torch.full((batch_size, 1), 1, dtype=torch.long, device=device)
    style_ids = torch.tensor(style_ids, device=device)
    
    for _ in range(max_length):
        with torch.no_grad():
            outputs = model(input_ids, style_ids)
            logits = outputs.logits[:, -1, :]
            
            # Sample next tokens
            probs = torch.softmax(logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
            
            # Append to sequences
            input_ids = torch.cat([input_ids, next_tokens], dim=1)
    
    # Return as list of individual tensors
    return [seq for seq in input_ids]
```

### GPU Memory Management

```python
def clear_gpu_cache():
    """Clear GPU cache to free memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def get_gpu_memory_info() -> dict:
    """Get GPU memory usage statistics."""
    if not torch.cuda.is_available():
        return {}
    
    return {
        'allocated_gb': torch.cuda.memory_allocated() / 1e9,
        'reserved_gb': torch.cuda.memory_reserved() / 1e9,
        'max_allocated_gb': torch.cuda.max_memory_allocated() / 1e9,
        'device_name': torch.cuda.get_device_name(0),
    }
```

---

## Evaluation Metrics

### Perplexity

```python
def calculate_perplexity(model, dataloader, device="cuda"):
    """Calculate model perplexity on dataset."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in dataloader:
            outputs = model(
                batch['input_ids'].to(device),
                batch['style_ids'].to(device),
                labels=batch['labels'].to(device)
            )
            
            # Accumulate loss
            total_loss += outputs.loss.item() * batch['input_ids'].size(0)
            total_tokens += batch['input_ids'].size(0)
    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))
    
    return perplexity.item()
```

### Pattern Quality Metrics

```python
def evaluate_pattern_quality(generated_pattern, reference_pattern):
    """
    Evaluate generated pattern quality.
    
    Metrics:
    - Note density similarity
    - Timing distribution similarity
    - Velocity distribution similarity
    - Rhythmic coherence
    """
    metrics = {}
    
    # 1. Note density
    gen_density = len(generated_pattern) / get_duration(generated_pattern)
    ref_density = len(reference_pattern) / get_duration(reference_pattern)
    metrics['density_similarity'] = 1 - abs(gen_density - ref_density) / ref_density
    
    # 2. Timing distribution (histogram comparison)
    gen_timings = [note['time'] % 480 for note in generated_pattern]
    ref_timings = [note['time'] % 480 for note in reference_pattern]
    
    import numpy as np
    gen_hist, _ = np.histogram(gen_timings, bins=16, range=(0, 480))
    ref_hist, _ = np.histogram(ref_timings, bins=16, range=(0, 480))
    
    # Normalize histograms
    gen_hist = gen_hist / gen_hist.sum()
    ref_hist = ref_hist / ref_hist.sum()
    
    # KL divergence
    kl_div = np.sum(ref_hist * np.log((ref_hist + 1e-10) / (gen_hist + 1e-10)))
    metrics['timing_similarity'] = 1 / (1 + kl_div)
    
    # 3. Overall quality score
    metrics['quality_score'] = np.mean([
        metrics['density_similarity'],
        metrics['timing_similarity'],
    ])
    
    return metrics
```

---

## Related Documents

- **Project Overview**: `.cursorcontext/01_project_overview.md`
- **Architecture**: `.cursorcontext/02_architecture.md`
- **Dependencies**: `.cursorcontext/03_dependencies.md`
- **MIDI Operations**: `.cursorcontext/04_midi_operations.md`
- **Common Tasks**: `.cursorcontext/06_common_tasks.md`