import torch
import torch.nn as nn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('tkagg')

def param_histogram(model:nn.Module, num_layers:int=50, range:tuple=(-2, 2)):
    """
    Plot histogram of the model parameters
    """
    params = []
    for param in model.parameters():
        if(len(params) <= num_layers):
            params.append(param.detach().cpu().numpy().flatten())
    params = torch.cat([torch.tensor(p) for p in params])

    plt.figure(figsize=(12, 8))
    plt.hist(params.numpy(), bins=50, range=range, color='blue', alpha=0.7)
    plt.title('Distribution of Model Parameters')
    plt.xlabel('Parameter Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

def grad_histogram(model:nn.Module):
    grads = [p.grad.cpu().numpy() for p in model.parameters() if p.grad is not None]
    plt.figure(figsize=(12, 8))
    for i, grad in enumerate(grads):
        plt.subplot(len(grads) // 2 + 1, 2, i + 1)
        plt.hist(grad.flatten(), bins=30, edgecolor='black')
        plt.title(f"Grad {i + 1} - {grad.shape}")
        plt.xlabel('Value')
        plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()