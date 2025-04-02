import os
import argparse
import yaml
import torch
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
import matplotlib.pyplot as plt

from models.lte_extractor import LTEExtractor
from models.attention_module import AttentionEmbedding
from models.hard_attention import HardAttention
from models.soft_attention import SoftAttention
from models.csfi import CSFI
from train import DehazeDecoder
from utils.metrics import get_metrics

def load_image(image_path, target_size=(256, 256)):
    """加载和预处理图像"""
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(img).unsqueeze(0)  # 添加批次维度

def inference(hazy_image_path, ground_truth_path, checkpoint_path, output_dir, config):
    """对一张图像进行去雾推理并评估"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载图像
    hazy_image = load_image(hazy_image_path, config['target_size']).cuda()
    
    # 如果有真实图像，加载它用于评估
    has_gt = ground_truth_path is not None
    if has_gt:
        ground_truth = load_image(ground_truth_path, config['target_size']).cuda()
    
    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lte_model = LTEExtractor().to(device)
    attention_model = AttentionEmbedding(scale_factor=1.0 / (256 ** 0.5)).to(device)
    hard_attention = HardAttention().to(device)
    soft_attention = SoftAttention(chunk_size=config['chunk_size']).to(device)
    csfi = CSFI(in_channels=256).to(device)
    decoder = DehazeDecoder().to(device)
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device)
    lte_model.load_state_dict(checkpoint['lte'])
    attention_model.load_state_dict(checkpoint['attention'])
    hard_attention.load_state_dict(checkpoint['hard_attention'])
    soft_attention.load_state_dict(checkpoint['soft_attention'])
    csfi.load_state_dict(checkpoint['csfi'])
    decoder.load_state_dict(checkpoint['decoder'])
    
    # 设置为评估模式
    lte_model.eval()
    attention_model.eval()
    hard_attention.eval()
    soft_attention.eval()
    csfi.eval()
    decoder.eval()
    
    # 执行推理
    with torch.no_grad():
        Q, K, V = lte_model(hazy_image)
        S, H = attention_model(Q, K, V)
        P = hard_attention(V, H)
        soft_output = soft_attention(P, S)
        features = csfi(soft_output, hazy_image)
        output = decoder(features)
    
        # 如果有真实图像，计算评估指标
        if has_gt:
            metrics = get_metrics()
            results = {}
            for name, metric_fn in metrics.items():
                results[name] = metric_fn(output, ground_truth).item()
            print("评估指标:")
            for name, value in results.items():
                print(f"  {name.upper()}: {value:.4f}")
    
    # 保存结果
    image_name = os.path.basename(hazy_image_path)
    output_path = os.path.join(output_dir, f"dehazed_{image_name}")
    
    # 将输出从[0,1]转换为[0,255]
    output_img = output.cpu().squeeze(0)
    output_img = output_img.clamp(0, 1)
    
    # 保存图像
    vutils.save_image(output_img, output_path)
    
    # 可视化
    n_images = 3 if has_gt else 2
    fig, axes = plt.subplots(1, n_images, figsize=(6*n_images, 6))
    
    # 还原归一化
    hazy_img = (hazy_image.cpu().squeeze(0) * 0.5 + 0.5).clamp(0, 1)
    
    # 显示原始有雾图像
    axes[0].imshow(hazy_img.permute(1, 2, 0))
    axes[0].set_title('Hazy Input')
    axes[0].axis('off')
    
    # 显示去雾结果
    axes[1].imshow(output_img.permute(1, 2, 0))
    axes[1].set_title('Dehazed Output')
    axes[1].axis('off')
    
    # 如果有真实图像，显示它
    if has_gt:
        gt_img = (ground_truth.cpu().squeeze(0) * 0.5 + 0.5).clamp(0, 1)
        axes[2].imshow(gt_img.permute(1, 2, 0))
        axes[2].set_title('Ground Truth')
        axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"comparison_{image_name}"))
    print(f"结果已保存到 {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="图像去雾推理")
    parser.add_argument("--image", required=True, help="有雾图像路径")
    parser.add_argument("--gt", default=None, help="真实图像路径 (可选，用于评估)")
    parser.add_argument("--checkpoint", required=True, help="模型检查点路径")
    parser.add_argument("--output", default="./results", help="输出目录")
    parser.add_argument("--config", default="./configs/training_config.yaml", help="配置文件")
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    inference(args.image, args.gt, args.checkpoint, args.output, config)