# 导入必要的库
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

# 导入配置文件
from config import Config

# 创建配置实例
config = Config()


class BasicLatBlk(nn.Module):
    """
    基础侧向连接块（Basic Lateral Block）
    
    侧向连接块用于在特征金字塔网络（FPN）中连接不同层级的特征。
    它通过1x1卷积来调整特征图的通道数，使得不同层级的特征能够进行有效的融合。
    
    这种设计常用于目标检测和语义分割任务中，用于：
    1. 调整特征图的通道数以匹配融合要求
    2. 在不改变空间分辨率的情况下进行特征变换
    3. 为后续的特征融合操作做准备
    """
    def __init__(self, in_channels=64, out_channels=64, inter_channels=64):
        """
        初始化基础侧向连接块
        
        Args:
            in_channels: 输入通道数，默认64
            out_channels: 输出通道数，默认64
            inter_channels: 中间层通道数，默认64（当前未使用，但保留用于扩展）
        """
        super(BasicLatBlk, self).__init__()
        # 根据配置设置中间层通道数：自适应模式下为输入通道数的1/4，否则为64
        # 注意：这里计算了inter_channels但实际未使用，可能是为了与其他模块保持一致的接口
        inter_channels = in_channels // 4 if config.dec_channels_inter == 'adap' else 64
        
        # 1x1卷积层，用于调整通道数
        # 不改变空间分辨率，只进行通道维度的变换
        self.conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入特征图，形状为 (N, in_channels, H, W)
            
        Returns:
            经过通道调整的特征图，形状为 (N, out_channels, H, W)
        """
        # 通过1x1卷积调整通道数
        x = self.conv(x)
        return x
