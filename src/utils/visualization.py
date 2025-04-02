import matplotlib.pyplot as plt
import torch

def visualize_attention(attn_tensor, save_path=None):
    """
    可视化注意力矩阵
    :param attn_tensor: [num_heads, H*W, H*W]
    """
    num_heads = attn_tensor.shape[0]
    
    plt.figure(figsize=(15, 5))
    for i in range(min(4, num_heads)):  # 最多显示4个头
        plt.subplot(1, 4, i+1)
        plt.imshow(attn_tensor[i].cpu().detach(), cmap='viridis')
        plt.title(f'Head {i}')
        plt.axis('off')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()