"""
Language Modeling Benchmark: Standard Attention vs Projection-Based Attention

Compares:
1. Baseline Transformer with standard multi-head self-attention
2. Hybrid Transformer with projection blocks + final standard attention

Benchmark on WikiText-2, IMDB, AG News, CMU Book Summaries datasets
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from transformers import GPT2Tokenizer
from datasets import load_dataset
import time
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')


# Data Loading Functions

def load_dataset_for_lm(dataset_name, tokenizer, max_length=512):
    """
    Generalized function to load text datasets for language modeling.
   
    Args:
        dataset_name: Name of the dataset ('wikitext-2', 'imdb', 'ag_news', 'cmu-book-summaries')
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
   
    Returns:
        tuple: (train_dataset, val_dataset, test_dataset) or (train_dataset, test_dataset)
    """
    dataset_configs = {
        'wikitext-2': {
            'path': 'wikitext',
            'name': 'wikitext-2-raw-v1',
            'text_column': 'text',
            'has_validation': True
        },
        'imdb': {
            'path': 'imdb',
            'name': None,
            'text_column': 'text',
            'has_validation': False
        },
        'ag_news': {
            'path': 'ag_news',
            'name': None,
            'text_column': 'text',
            'has_validation': False
        },
        'cmu-book-summaries': {
            'path': 'textminr/cmu-book-summaries',
            'name': None,
            'text_column': 'summary',
            'has_validation': False
        }
    }
   
    if dataset_name not in dataset_configs:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
   
    config = dataset_configs[dataset_name]
   
    def tokenize_function(examples):
        text_column = config['text_column']
        input_ids = []
        attention_masks = []
       
        for text in examples[text_column]:
            if text and text.strip():
                tokenized = tokenizer(text, truncation=True, padding=False, max_length=max_length)
                input_ids.append(tokenized['input_ids'])
                attention_masks.append(tokenized['attention_mask'])
            else:
                input_ids.append([])
                attention_masks.append([])
       
        return {'input_ids': input_ids, 'attention_mask': attention_masks}
   
    # Load datasets
    if config['has_validation']:
        train_ds = load_dataset(config['path'], config['name'], split='train')
        val_ds = load_dataset(config['path'], config['name'], split='validation')
        test_ds = load_dataset(config['path'], config['name'], split='test')
        datasets = {'train': train_ds, 'validation': val_ds, 'test': test_ds}
    else:
        train_ds = load_dataset(config['path'], config['name'], split='train')
        test_ds = load_dataset(config['path'], config['name'], split='test')
        datasets = {'train': train_ds, 'test': test_ds}
   
    # Tokenize and filter
    tokenized_datasets = {}
    for split_name, dataset in datasets.items():
        columns_to_remove = [col for col in dataset.column_names if col != config['text_column']]
        tokenized = dataset.map(tokenize_function, batched=True, remove_columns=columns_to_remove)
        tokenized = tokenized.filter(lambda x: len(x['input_ids']) > 1)
        tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask'])
        tokenized_datasets[split_name] = tokenized
   
    # Return appropriate splits
    if config['has_validation']:
        return tokenized_datasets['train'], tokenized_datasets['validation'], tokenized_datasets['test']
    else:
        return tokenized_datasets['train'], tokenized_datasets['test']


def load_wikitext2(tokenizer, max_length=512):
    """Load WikiText-2 dataset"""
    return load_dataset_for_lm('wikitext-2', tokenizer, max_length)


def load_imdb(tokenizer, max_length=512):
    """Load IMDB reviews dataset"""
    return load_dataset_for_lm('imdb', tokenizer, max_length)


def load_ag_news(tokenizer, max_length=512):
    """Load AG News dataset"""
    return load_dataset_for_lm('ag_news', tokenizer, max_length)


def load_cmu_book_summaries(tokenizer, max_length=512, split_data=True, val_size=0.1, test_size=0.1):
    """
    Load CMU Book Summaries dataset with optional train/val/test split.
   
    Args:
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        split_data: Whether to split into train/val/test
        val_size: Fraction for validation
        test_size: Fraction for test
   
    Returns:
        (train_data, val_data, test_data) if split_data=True, else train_data
    """
    dataset_config = {
        'path': 'textminr/cmu-book-summaries',
        'name': None,
        'text_column': 'summary'
    }
   
    def tokenize_function(examples):
        input_ids = []
        attention_masks = []
       
        for text in examples[dataset_config['text_column']]:
            if text and text.strip():
                tokenized = tokenizer(text, truncation=True, padding=False, max_length=max_length)
                input_ids.append(tokenized['input_ids'])
                attention_masks.append(tokenized['attention_mask'])
            else:
                input_ids.append([])
                attention_masks.append([])
       
        return {'input_ids': input_ids, 'attention_mask': attention_masks}
   
    # Load dataset
    train_data = load_dataset(dataset_config['path'], dataset_config['name'], split='train')
   
    # Tokenize
    columns_to_remove = [col for col in train_data.column_names if col != dataset_config['text_column']]
    train_data = train_data.map(tokenize_function, batched=True, remove_columns=columns_to_remove)
    train_data = train_data.filter(lambda x: len(x['input_ids']) > 1)
    train_data.set_format(type='torch', columns=['input_ids', 'attention_mask'])
   
    if not split_data:
        return train_data
   
    # Split into train/val/test
    test_split = train_data.train_test_split(test_size=test_size, shuffle=True, seed=42)
    train_data = test_split['train']
    test_data = test_split['test']
   
    val_ratio = val_size / (1 - test_size)
    train_val_split = train_data.train_test_split(test_size=val_ratio, shuffle=True, seed=42)
    train_data = train_val_split['train']
    val_data = train_val_split['test']
   
    return train_data, val_data, test_data


def collate_fn(batch, pad_token_id=50256):
    """Pad sequences to max length in batch"""
    max_length = max(len(item["input_ids"]) for item in batch)
   
    input_ids = []
    attention_masks = []
   
    for item in batch:
        padded_input = torch.cat([
            item["input_ids"],
            torch.full((max_length - len(item["input_ids"]),), pad_token_id, dtype=torch.long)
        ])
        input_ids.append(padded_input)
       
        padded_mask = torch.cat([
            item["attention_mask"],
            torch.zeros(max_length - len(item["attention_mask"]), dtype=torch.long)
        ])
        attention_masks.append(padded_mask)
   
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_masks)
    }


# Model Components

class StandardAttentionBlock(nn.Module):
    """Standard transformer block with multi-head causal self-attention"""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, causal_mask=None):
        # Self-attention with causal mask
        attn_out, _ = self.attention(x, x, x, attn_mask=causal_mask, need_weights=False)
        x = self.norm1(x + attn_out)
       
        # Feedforward
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class ProjectionBlock(nn.Module):
    """Projection-based block (no attention mechanism)"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False),
            nn.GELU(),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, causal_mask=None):
        # Projection (mask not used here, but kept for interface consistency)
        proj_out = self.proj(x)
        x = self.norm1(x + proj_out)
       
        # Feedforward
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


# Models

class BaselineTransformer(nn.Module):
    """Transformer with standard multi-head attention in all layers"""
    def __init__(self, vocab_size, d_model=512, n_heads=8, n_layers=4,
                 d_ff=2048, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
       
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
       
        self.blocks = nn.ModuleList([
            StandardAttentionBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
       
        self.final_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)
       
        # Register buffer for causal mask
        causal_mask = torch.triu(torch.ones(max_seq_len, max_seq_len) * float('-inf'), diagonal=1)
        self.register_buffer('causal_mask', causal_mask)
       
    def forward(self, input_ids):
        batch_size, seq_len = input_ids.size()
       
        # Embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.pos_embedding(positions)
       
        # Get causal mask for this sequence length
        mask = self.causal_mask[:seq_len, :seq_len]
       
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, causal_mask=mask)
       
        x = self.final_norm(x)
        logits = self.output_projection(x)
       
        return logits


class HybridTransformer(nn.Module):
    """Transformer with configurable projection/attention blocks"""
    def __init__(self, vocab_size, d_model=512, n_heads=8, n_layers=4,
                 d_ff=2048, max_seq_len=512, dropout=0.1,
                 use_projection=[True, True, False], pos_encoding_bias=10.0):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.pos_encoding_bias = pos_encoding_bias
       
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
       
        # Ensure use_projection list matches n_layers
        if len(use_projection) != n_layers:
            raise ValueError(f"use_projection list length ({len(use_projection)}) must match n_layers ({n_layers})")
       
        # Build layers based on use_projection configuration
        self.blocks = nn.ModuleList()
        for is_projection in use_projection:
            if is_projection:
                self.blocks.append(ProjectionBlock(d_model, d_ff, dropout))
            else:
                self.blocks.append(StandardAttentionBlock(d_model, n_heads, d_ff, dropout))
       
        self.final_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)
       
        # Register buffer for causal mask
        causal_mask = torch.triu(torch.ones(max_seq_len, max_seq_len) * float('-inf'), diagonal=1)
        self.register_buffer('causal_mask', causal_mask)
        self.use_projection = use_projection
       
    def forward(self, input_ids):
        batch_size, seq_len = input_ids.size()
       
        # Embeddings (using multiplication for positional encoding)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) * (self.pos_embedding(positions) + self.pos_encoding_bias)
       
        # Get causal mask for this sequence length
        mask = self.causal_mask[:seq_len, :seq_len]
       
        # Apply blocks with appropriate masking
        for block, is_projection in zip(self.blocks, self.use_projection):
            if is_projection:
                x = block(x, causal_mask=None)  # No mask needed for projection
            else:
                x = block(x, causal_mask=mask)  # Apply causal mask for attention
       
        x = self.final_norm(x)
        logits = self.output_projection(x)
       
        return logits


# Training & Evaluation

def train_epoch(model, dataloader, optimizer, device, pad_token_id):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
   
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
       
        # Create targets (next token prediction)
        targets = input_ids[:, 1:].contiguous()
        inputs = input_ids[:, :-1].contiguous()
       
        # Mask to ignore padding tokens
        mask = (targets != pad_token_id)
       
        optimizer.zero_grad()
       
        logits = model(inputs)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=pad_token_id
        )
       
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
       
        total_loss += loss.item()
       
        # Calculate accuracy excluding padding
        predicted = logits.argmax(dim=-1)
        correct += ((predicted == targets) & mask).sum().item()
        total += mask.sum().item()
   
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total if total > 0 else 0
    perplexity = torch.exp(torch.tensor(avg_loss))
   
    return avg_loss, accuracy, perplexity.item()


def validate(model, dataloader, device, pad_token_id):
    """Validate model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
   
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", leave=False):
            input_ids = batch["input_ids"].to(device)
           
            targets = input_ids[:, 1:].contiguous()
            inputs = input_ids[:, :-1].contiguous()
           
            mask = (targets != pad_token_id)
           
            logits = model(inputs)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=pad_token_id
            )
           
            total_loss += loss.item()
           
            predicted = logits.argmax(dim=-1)
            correct += ((predicted == targets) & mask).sum().item()
            total += mask.sum().item()
   
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total if total > 0 else 0
    perplexity = torch.exp(torch.tensor(avg_loss))
   
    return avg_loss, accuracy, perplexity.item()


def measure_inference_speed(model, loader, device, warmup=2, reps=3):
    """Measure inference speed in tokens/sec"""
    model.eval()
   
    # Warmup
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= warmup:
                break
            input_ids = batch["input_ids"].to(device)
            _ = model(input_ids[:, :-1])
   
    # Measure
    total_time = 0.0
    total_tokens = 0
   
    with torch.no_grad():
        for r in range(reps):
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                inputs = input_ids[:, :-1]
               
                start = time.time()
                _ = model(inputs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                total_time += time.time() - start
               
                total_tokens += inputs.numel()
   
    return total_tokens / total_time


# Main Experiment

def run_experiment(dataset_name, train_loader, val_loader, config, device, pad_token_id):
    """Run experiment for one dataset"""
    print("\n" + "="*70)
    print(f"Dataset: {dataset_name}")
    print("="*70)
   
    results = {'dataset': dataset_name}
    vocab_size = config['vocab_size']
   
    models_to_test = [
        ("Baseline Transformer", lambda: BaselineTransformer(
            vocab_size=vocab_size,
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            n_layers=config['n_layers'],
            d_ff=config['d_ff'],
            max_seq_len=config['max_seq_len'],
            dropout=config['dropout']
        )),
        ("Hybrid Transformer", lambda: HybridTransformer(
            vocab_size=vocab_size,
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            n_layers=config['n_layers'],
            d_ff=config['d_ff'],
            max_seq_len=config['max_seq_len'],
            dropout=config['dropout'],
            use_projection=config.get('use_projection', [True, True, False]),
            pos_encoding_bias=config.get('pos_encoding_bias', 10.0)
        ))
    ]
   
    for idx, (model_name, model_fn) in enumerate(models_to_test, 1):
        print(f"\n[{idx}/{len(models_to_test)}] Training {model_name}...")
       
        model = model_fn().to(device)
        print(f"{model_name} params: {sum(p.numel() for p in model.parameters()):,}")
       
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
       
        # Training loop
        start_time = time.time()
        best_val_loss = float('inf')
       
        for epoch in range(config['n_epochs']):
            train_loss, train_acc, train_ppl = train_epoch(
                model, train_loader, optimizer, device, pad_token_id
            )
            val_loss, val_acc, val_ppl = validate(
                model, val_loader, device, pad_token_id
            )
           
            best_val_loss = min(best_val_loss, val_loss)
           
            if (epoch + 1) % 2 == 0:
                print(f"Epoch {epoch+1}: Train PPL={train_ppl:.2f}, Val PPL={val_ppl:.2f}, "
                      f"Val Acc={val_acc:.4f}")
       
        train_time = time.time() - start_time
       
        # Final evaluation
        val_loss, val_acc, val_ppl = validate(model, val_loader, device, pad_token_id)
        speed = measure_inference_speed(model, val_loader, device)
       
        prefix = model_name.lower().replace(' ', '_')
        results.update({
            f'{prefix}_ppl': val_ppl,
            f'{prefix}_acc': val_acc,
            f'{prefix}_loss': val_loss,
            f'{prefix}_speed': speed,
            f'{prefix}_time': train_time
        })
       
        print(f"\n{model_name} -> PPL: {val_ppl:.2f}, Acc: {val_acc:.4f}, "
              f"Speed: {speed:.0f} tokens/s")
       
        # Clean up
        del model, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
   
    return results


# Main

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
   
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = tokenizer.pad_token_id
   
    CONFIG = {
        'vocab_size': tokenizer.vocab_size,
        'max_seq_len': 256,
        'd_model': 512,
        'n_heads': 8,
        'n_layers': 3,
        'd_ff': 1024,
        'batch_size': 32,
        'n_epochs': 14,
        'lr': 1e-4,
        'dropout': 0.15,
        'use_projection': [True, True, False],  # Configure projection/attention layers
        'pos_encoding_bias': 10.0,  # Positional encoding bias for hybrid model
        'use_full_dataset': False,  # If True, use full dataset; if False, use subset
        'train_subset_size': 5000,  # Used when use_full_dataset=False
        'val_subset_size': 1000     # Used when use_full_dataset=False
    }
   
    # Dataset configurations
    datasets_config = [
        # ("WikiText-2", lambda: load_wikitext2(tokenizer, max_length=CONFIG['max_seq_len'])),
        # Uncomment to test other datasets:
        # ("IMDB", lambda: load_imdb(tokenizer, max_length=CONFIG['max_seq_len'])),
         ("AG News", lambda: load_ag_news(tokenizer, max_length=CONFIG['max_seq_len'])),
        # ("CMU Book Summaries", lambda: load_cmu_book_summaries(
        #     tokenizer, max_length=CONFIG['max_seq_len'], split_data=True
        # )),
    ]
   
    results = []
   
    for dataset_name, load_fn in datasets_config:
        print(f"\nLoading {dataset_name}...")
       
        # Load dataset
        dataset_output = load_fn()
        if len(dataset_output) == 3:
            train_dataset, val_dataset, _ = dataset_output
        else:
            train_dataset, val_dataset = dataset_output
       
        print(f"{dataset_name} - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
       
        # Use subset or full dataset based on configuration
        if CONFIG['use_full_dataset']:
            train_subset = train_dataset
            val_subset = val_dataset
            print(f"Using full dataset - Train: {len(train_subset)}, Val: {len(val_subset)}")
        else:
            train_subset_size = min(CONFIG['train_subset_size'], len(train_dataset))
            val_subset_size = min(CONFIG['val_subset_size'], len(val_dataset))
           
            train_subset = Subset(train_dataset, range(train_subset_size))
            val_subset = Subset(val_dataset, range(val_subset_size))
           
            print(f"Using subset - Train: {len(train_subset)}, Val: {len(val_subset)}")
       
        # Create dataloaders
        train_loader = DataLoader(
            train_subset,
            batch_size=CONFIG['batch_size'],
            shuffle=True,
            collate_fn=lambda batch: collate_fn(batch, pad_token_id)
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=CONFIG['batch_size'],
            shuffle=False,
            collate_fn=lambda batch: collate_fn(batch, pad_token_id)
        )
       
        # Run experiment
        result = run_experiment(
            dataset_name, train_loader, val_loader, CONFIG, device, pad_token_id
        )
        results.append(result)
   
    # Print summary
    print("\n" + "="*70)
    print("Final Benchmark Summary")
    print("="*70)
    for r in results:
        print(f"\n{r['dataset']}:")
        print(f"  Baseline Transformer: PPL={r['baseline_transformer_ppl']:.2f}, "
              f"Acc={r['baseline_transformer_acc']:.4f}, "
              f"Speed={r['baseline_transformer_speed']:.0f} tokens/s")
        print(f"  Hybrid Transformer:   PPL={r['hybrid_transformer_ppl']:.2f}, "
              f"Acc={r['hybrid_transformer_acc']:.4f}, "
              f"Speed={r['hybrid_transformer_speed']:.0f} tokens/s")
       
        ppl_improvement = ((r['baseline_transformer_ppl'] - r['hybrid_transformer_ppl']) /
                          r['baseline_transformer_ppl']) * 100
        speedup = r['hybrid_transformer_speed'] / r['baseline_transformer_speed']
        print(f"  -> Hybrid vs Baseline: PPL {ppl_improvement:+.1f}%, Speed {speedup:.2f}x")
	
