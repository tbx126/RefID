#src/dehaze_dataset.py

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from glob import glob

class DehazeDataset(Dataset):
    """保持长宽比的padding数据集实现"""
    def __init__(self, dehazed1_dir, dehazed2_dir, clear_dir, target_size=(256, 256)):
        self.target_size = target_size
        self.pad_value = [0.7, 0.7, 0.7]
        self.file_triplets = self._match_files(dehazed1_dir, dehazed2_dir, clear_dir)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    
    def _match_files(self, dir1, dir2, dir3):
        """匹配三个目录中的对应文件"""
        files1 = sorted(glob(os.path.join(dir1, '*.png')))
        files2 = sorted(glob(os.path.join(dir2, '*.png')))
        files3 = sorted(glob(os.path.join(dir3, '*.png')))
        assert len(files1) == len(files2) == len(files3)
        return list(zip(files1, files2, files3))
    
    def _smart_pad(self, img):
        """智能padding保持长宽比"""
        w, h = img.size
        scale = min(self.target_size[1]/w, self.target_size[0]/h)
        new_w, new_h = int(w*scale), int(h*scale)
        img = img.resize((new_w, new_h), Image.BICUBIC)
        padded_img = Image.new('RGB', (self.target_size[1], self.target_size[0]), 
                             color=tuple(int(255*x) for x in self.pad_value))
        paste_pos = ((self.target_size[1]-new_w)//2, (self.target_size[0]-new_h)//2)
        padded_img.paste(img, paste_pos)
        return padded_img
    
    def __len__(self):
        return len(self.file_triplets)
    
    def __getitem__(self, idx):
        img1 = Image.open(self.file_triplets[idx][0]).convert('RGB')
        img2 = Image.open(self.file_triplets[idx][1]).convert('RGB')
        img3 = Image.open(self.file_triplets[idx][2]).convert('RGB')
        
        img1 = self._smart_pad(img1)
        img2 = self._smart_pad(img2)
        img3 = self._smart_pad(img3)
        
        return {
            'dehazed1': self.normalize(self.to_tensor(img1)),
            'dehazed2': self.normalize(self.to_tensor(img2)),
            'clear': self.normalize(self.to_tensor(img3))
        }