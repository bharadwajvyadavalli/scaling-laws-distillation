import matplotlib.pyplot as plt
import numpy as np
import json
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def load_results(file_path='benchmark_results.json'):
    """Load benchmark results from JSON"""
    with open(file_path, 'r') as f:
        return json.load(f)

def plot_scaling_curves(results):
    """Main scaling law visualizations"""
    # Extract data
    params = [r['parameters']['total_M'] for r in results]
    perplexity = [r['perplexity'] for r in results]
    flops = [r['flops']['flops_G'] for r in results]
    latency = [r['speed']['avg_latency_ms'] for r in results]
    
    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Parameters vs Perplexity (main scaling law)
    ax = axes[0, 0]
    ax.scatter(params, perplexity, s=100, alpha=0.7)
    ax.plot(params, perplexity, 'b--', alpha=0.5)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Parameters (M)')
    ax.set_ylabel('Perplexity')
    ax.set_title('Scaling Law: Model Size vs Performance')
    ax.grid(True, alpha=0.3)
    
    # Add power law fit
    log_params = np.log(params)
    log_perp = np.log(perplexity)
    z = np.polyfit(log_params, log_perp, 1)
    p = np.poly1d(z)
    ax.plot(params, np.exp(p(log_params)), 'r-', alpha=0.5, 
            label=f'Power law: PPL ∝ Params^{z[0]:.2f}')
    ax.legend()
    
    # 2. FLOPs vs Perplexity
    ax = axes[0, 1]
    ax.scatter(flops, perplexity, s=100, alpha=0.7, c='green')
    ax.plot(flops, perplexity, 'g--', alpha=0.5)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('FLOPs (G)')
    ax.set_ylabel('Perplexity')
    ax.set_title('Compute Efficiency')
    ax.grid(True, alpha=0.3)
    
    # 3. Parameters vs Latency
    ax = axes[1, 0]
    ax.scatter(params, latency, s=100, alpha=0.7, c='orange')
    ax.plot(params, latency, 'orange', linestyle='--', alpha=0.5)
    ax.set_xscale('log')
    ax.set_xlabel('Parameters (M)')
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Model Size vs Inference Speed')
    ax.grid(True, alpha=0.3)
    
    # 4. Efficiency Frontier (Perplexity/Parameter)
    ax = axes[1, 1]
    efficiency = [p/param for p, param in zip(perplexity, params)]
    ax.bar(range(len(results)), efficiency, color='purple', alpha=0.7)
    ax.set_xlabel('Model Variant')
    ax.set_ylabel('Perplexity / Parameter (M)')
    ax.set_title('Parameter Efficiency')
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels([f"{p:.1f}M" for p in params], rotation=45)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Knowledge Distillation Scaling Analysis', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('scaling_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig

def plot_distillation_comparison(teacher_results, student_results):
    """Compare teacher vs distilled student performance"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Prepare data
    categories = ['Teacher', 'Student']
    
    # Performance comparison
    ax = axes[0]
    metrics = ['Perplexity', 'Parameters (M)', 'FLOPs (G)']
    teacher_vals = [
        teacher_results.get('perplexity', 100),
        teacher_results.get('parameters', {}).get('total_M', 100),
        teacher_results.get('flops', {}).get('flops_G', 100)
    ]
    student_vals = [
        student_results.get('perplexity', 50),
        student_results.get('parameters', {}).get('total_M', 10),
        student_results.get('flops', {}).get('flops_G', 10)
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax.bar(x - width/2, teacher_vals, width, label='Teacher', alpha=0.8)
    ax.bar(x + width/2, student_vals, width, label='Student', alpha=0.8)
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Value')
    ax.set_title('Teacher vs Student Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_yscale('log')
    
    # Compression ratio
    ax = axes[1]
    compression_metrics = {
        'Size Reduction': teacher_vals[1] / student_vals[1],
        'FLOPs Reduction': teacher_vals[2] / student_vals[2],
        'Perplexity Ratio': student_vals[0] / teacher_vals[0]
    }
    
    bars = ax.bar(compression_metrics.keys(), compression_metrics.values(), 
                   color=['blue', 'green', 'red'], alpha=0.7)
    ax.set_ylabel('Ratio')
    ax.set_title('Compression Metrics')
    ax.axhline(y=1, color='black', linestyle='--', alpha=0.5)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}x', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('distillation_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig

def plot_data_scaling(data_sizes, perplexities):
    """Plot how data size affects distillation"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.scatter(data_sizes, perplexities, s=100, alpha=0.7, c='teal')
    ax.plot(data_sizes, perplexities, 'teal', linestyle='--', alpha=0.5)
    
    ax.set_xscale('log')
    ax.set_xlabel('Training Data Size (tokens)')
    ax.set_ylabel('Perplexity')
    ax.set_title('Data Scaling: Effect of Training Data Size on Distillation')
    ax.grid(True, alpha=0.3)
    
    # Fit and plot trend
    log_data = np.log(data_sizes)
    z = np.polyfit(log_data, perplexities, 1)
    p = np.poly1d(z)
    ax.plot(data_sizes, p(log_data), 'r-', alpha=0.5,
            label=f'Trend: {z[0]:.3f} * log(data) + {z[1]:.2f}')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('data_scaling.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig

if __name__ == "__main__":
    # Load results
    results = load_results()
    
    # Generate all plots
    print("Generating scaling curves...")
    plot_scaling_curves(results)
    
    # Example: compare teacher vs student
    if len(results) >= 2:
        print("Generating comparison plots...")
        plot_distillation_comparison(results[0], results[1])
    
    print("Plots saved as PNG files.")