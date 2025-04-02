#src/main.py

import yaml
import torch
from torch.utils.data import DataLoader
from data.dehaze_dataset import DehazeDataset
from models.lte_extractor import LTEExtractor
from models.attention_module import AttentionEmbedding

def main():
    # 加载配置
    with open('./configs/dataset_config.yaml') as f:
        config = yaml.safe_load(f)
    
    # 初始化数据集
    dataset = DehazeDataset(
        dehazed1_dir=config['data_paths']['dehazed1'],
        dehazed2_dir=config['data_paths']['dehazed2'],
        clear_dir=config['data_paths']['clear'],
        target_size=config['target_size']
    )

    print("Dataset size:", len(dataset))
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4
    )

    batch = next(iter(dataloader))
    print("Batch keys:", batch.keys())
    print("Dehazed1 shape:", batch['dehazed1'].shape)
    print("Dehazed2 shape:", batch['dehazed2'].shape)
    print("Clear shape:", batch['clear'].shape)
    
    # 初始化模型
    lte_model = LTEExtractor().cuda()
    attention_model = AttentionEmbedding(scale_factor=1.0 / (256 ** 0.5)).cuda()

    print("Model device:", next(lte_model.parameters()).device)
    print("LTE model architecture:\n", lte_model)
    print("Attention model architecture:\n", attention_model)
    
    # 训练循环示例
    for batch in dataloader:
        # 将数据移动到GPU
        dehazed1 = batch['dehazed1'].cuda()
        dehazed2 = batch['dehazed2'].cuda()
        clear = batch['clear'].cuda()
        
        # 提取特征
        Q, K, V = lte_model(dehazed1)
        
        # 执行注意力嵌入
        S, H = attention_model(Q, K, V)
        
        print("Attention score shape:", S.shape)
        print("Attention output shape:", H.shape)
        
        # ... 后续训练逻辑
        break  # 仅执行一次循环进行测试

if __name__ == '__main__':
    main()
