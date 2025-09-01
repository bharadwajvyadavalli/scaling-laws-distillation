import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
import json
from models import TinyStudent, count_parameters, estimate_flops
import time

def evaluate_perplexity(model, dataloader, device='cpu'):
    """Calculate perplexity on dataset"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            
            # Shift for language modeling
            labels = input_ids[:, 1:].contiguous()
            outputs = model(input_ids[:, :-1])
            logits = outputs.logits
            
            # Calculate loss
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                reduction='sum'
            )
            
            total_loss += loss.item()
            total_tokens += labels.numel()
    
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    
    return perplexity, avg_loss

def evaluate_speed(model, input_shape=(1, 128), device='cpu', num_runs=100):
    """Measure inference speed"""
    model.eval()
    model.to(device)
    
    # Warmup
    dummy_input = torch.randint(0, 1000, input_shape).to(device)
    for _ in range(10):
        _ = model(dummy_input)
    
    # Measure
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start = time.time()
    
    for _ in range(num_runs):
        _ = model(dummy_input)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end = time.time()
    
    avg_time = (end - start) / num_runs
    throughput = input_shape[0] / avg_time  # samples/second
    
    return {
        'avg_latency_ms': avg_time * 1000,
        'throughput': throughput,
        'device': str(device)
    }

def benchmark_model(model_path, data_percent=10):
    """Full benchmark suite for a model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    # Recreate model
    model = TinyStudent(
        hidden_size=config['student_hidden_size'],
        num_layers=config['student_layers']
    ).to(device)
    model.load_state_dict(checkpoint['model_state'])
    
    # Load test data
    dataset = load_dataset("wikitext", "wikitext-2-v1", split=f"test[:{data_percent}%]")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    def tokenize(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)
    
    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format(type='torch', columns=['input_ids'])
    dataloader = DataLoader(dataset, batch_size=32)
    
    # Evaluate
    perplexity, loss = evaluate_perplexity(model, dataloader, device)
    speed_metrics = evaluate_speed(model, device=device)
    param_count = count_parameters(model)
    flops = estimate_flops(model)
    
    results = {
        'model_path': model_path,
        'perplexity': perplexity,
        'loss': loss,
        'parameters': param_count,
        'flops': flops,
        'speed': speed_metrics,
        'config': config
    }
    
    return results

def compare_models(model_paths):
    """Compare multiple models and save results"""
    all_results = []
    
    for path in model_paths:
        print(f"Evaluating {path}...")
        results = benchmark_model(path)
        all_results.append(results)
        
        print(f"  Perplexity: {results['perplexity']:.2f}")
        print(f"  Parameters: {results['parameters']['total_M']:.2f}M")
        print(f"  Latency: {results['speed']['avg_latency_ms']:.2f}ms")
        print()
    
    # Save results
    with open('benchmark_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print("Results saved to benchmark_results.json")
    return all_results

if __name__ == "__main__":
    # Example: evaluate a single model
    model_path = "student_256h_4l.pt"
    results = benchmark_model(model_path)
    
    print("Benchmark Results:")
    print("-" * 50)
    print(f"Perplexity: {results['perplexity']:.2f}")
    print(f"Loss: {results['loss']:.4f}")
    print(f"Parameters: {results['parameters']['total_M']:.2f}M")
    print(f"FLOPs: {results['flops']['flops_G']:.2f}G")
    print(f"Latency: {results['speed']['avg_latency_ms']:.2f}ms")
    print(f"Throughput: {results['speed']['throughput']:.2f} samples/s")