import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
import yaml
from models import TinyStudent

def distill_step(teacher, student, batch, temperature=4.0):
    """Single distillation step"""
    kl_loss = nn.KLDivLoss(reduction='batchmean')
    
    # Teacher forward (no grad)
    with torch.no_grad():
        teacher_logits = teacher(batch['input_ids']).logits
    
    # Student forward
    student_logits = student(batch['input_ids']).logits
    
    # Distillation loss
    loss = kl_loss(
        torch.log_softmax(student_logits / temperature, dim=-1),
        torch.softmax(teacher_logits / temperature, dim=-1)
    ) * (temperature ** 2)
    
    return loss

def train_distillation(config_path="config.yaml"):
    """Main training loop"""
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load teacher model
    teacher = AutoModel.from_pretrained(config['teacher_model']).to(device)
    teacher.eval()
    
    # Initialize student model
    if config['student_type'] == 'tiny':
        student = TinyStudent(
            hidden_size=config['student_hidden_size'],
            num_layers=config['student_layers']
        ).to(device)
    else:
        student = AutoModel.from_pretrained(config['student_model']).to(device)
    
    # Load dataset
    dataset = load_dataset("wikitext", "wikitext-2-v1", split=f"train[:{config['data_percent']}%]")
    tokenizer = AutoTokenizer.from_pretrained(config['teacher_model'])
    
    def tokenize(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)
    
    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    
    # Optimizer
    optimizer = torch.optim.Adam(student.parameters(), lr=config['learning_rate'])
    
    # Training loop
    student.train()
    for epoch in range(config['epochs']):
        total_loss = 0
        for i, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            loss = distill_step(teacher, student, batch, config['temperature'])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if i % 100 == 0:
                print(f"Epoch {epoch}, Step {i}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} completed. Average Loss: {avg_loss:.4f}")
    
    # Save model
    torch.save({
        'model_state': student.state_dict(),
        'config': config,
        'final_loss': avg_loss
    }, f"student_{config['student_hidden_size']}h_{config['student_layers']}l.pt")
    
    print(f"Training complete. Model saved.")

if __name__ == "__main__":
    train_distillation()