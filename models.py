import torch
import torch.nn as nn

class TinyStudent(nn.Module):
    """Configurable tiny transformer for scaling experiments"""
    def __init__(self, vocab_size=30522, hidden_size=256, num_layers=4, num_heads=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Embeddings
        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(512, hidden_size)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 4,
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        
        # Output
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, input_ids, **kwargs):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)
        
        # Embeddings
        x = self.embeddings(input_ids) + self.position_embeddings(position_ids)
        
        # Transformer layers
        for layer in self.layers:
            x = layer(x)
        
        # Output
        x = self.norm(x)
        logits = self.head(x)
        
        # Return in HuggingFace format
        return type('Output', (), {'logits': logits})()

def count_parameters(model):
    """Count total and trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total': total,
        'trainable': trainable,
        'total_M': total / 1e6,
        'trainable_M': trainable / 1e6
    }

def estimate_flops(model, seq_length=128, batch_size=1):
    """Rough FLOPs estimation for transformer models"""
    params = count_parameters(model)['total']
    
    # Simplified: FLOPs ≈ 2 * params * seq_length * batch_size
    # (forward pass, ignoring attention complexity)
    flops = 2 * params * seq_length * batch_size
    
    # Add attention FLOPs (quadratic in seq_length)
    if hasattr(model, 'num_layers'):
        num_layers = model.num_layers
        hidden_size = model.hidden_size
        # Attention: O(seq_length^2 * hidden_size)
        attention_flops = num_layers * seq_length * seq_length * hidden_size * batch_size
        flops += attention_flops
    
    return {
        'flops': flops,
        'flops_G': flops / 1e9,
        'formula': '2*params*seq_len + layers*seq_len^2*hidden'
    }

def create_model_variants():
    """Create different sized models for scaling experiments"""
    variants = [
        {'name': 'tiny', 'hidden_size': 128, 'num_layers': 2},
        {'name': 'small', 'hidden_size': 256, 'num_layers': 4},
        {'name': 'medium', 'hidden_size': 512, 'num_layers': 6},
        {'name': 'base', 'hidden_size': 768, 'num_layers': 12},
    ]
    
    models = {}
    for v in variants:
        model = TinyStudent(
            hidden_size=v['hidden_size'],
            num_layers=v['num_layers'],
            num_heads=max(4, v['hidden_size'] // 64)
        )
        models[v['name']] = {
            'model': model,
            'params': count_parameters(model),
            'flops': estimate_flops(model),
            'config': v
        }
    
    return models

if __name__ == "__main__":
    # Test model creation and stats
    models = create_model_variants()
    
    print("Model Variants:")
    print("-" * 50)
    for name, info in models.items():
        print(f"{name.upper()}:")
        print(f"  Parameters: {info['params']['total_M']:.2f}M")
        print(f"  FLOPs: {info['flops']['flops_G']:.2f}G")
        print(f"  Config: {info['config']}")
        print()