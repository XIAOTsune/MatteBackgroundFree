# 导入PyTorch神经网络模块
import torch.nn as nn
# 导入工具函数用于构建激活层和归一化层
from models.modules.utils import build_act_layer, build_norm_layer


class StemLayer(nn.Module):
    """
    Stem层（词干层）
    
    这是一个用于预处理输入特征的词干层，通常用于网络的开始部分。
    它通过两个卷积层将输入特征从原始通道数转换到目标通道数，
    并在每个卷积层后应用归一化和激活函数。
    
    这个设计参考了InternImage的Stem层结构，用于在进入主干网络之前
    对输入进行初步的特征提取和通道调整。
    
    Args:
        in_channels (int): 输入通道数，默认为4（RGB + 1个额外通道）
        inter_channels (int): 中间层通道数，用于第一个卷积层的输出
        out_channels (int): 输出通道数，最终输出的特征通道数
        act_layer (str): 激活函数类型，支持 'ReLU', 'SiLU', 'GELU' 等
        norm_layer (str): 归一化层类型，支持 'BN' (BatchNorm) 和 'LN' (LayerNorm)
    """

    def __init__(self,
                 in_channels=3+1,
                 inter_channels=48,
                 out_channels=96,
                 act_layer='GELU',
                 norm_layer='BN'):
        """
        初始化StemLayer
        
        Args:
            in_channels (int): 输入通道数，默认为4（RGB + 1个额外通道）
            inter_channels (int): 中间层通道数，默认为48
            out_channels (int): 输出通道数，默认为96
            act_layer (str): 激活函数类型，默认为'GELU'
            norm_layer (str): 归一化层类型，默认为'BN'
        """
        super().__init__()
        
        # 第一个卷积层：将输入通道数转换为中间通道数
        # 使用3x3卷积核，步长为1，填充为1，保持空间尺寸不变
        self.conv1 = nn.Conv2d(in_channels,
                               inter_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        
        # 第一个归一化层：对中间特征进行归一化
        # 输入输出格式都是channels_first (N, C, H, W)
        self.norm1 = build_norm_layer(
            inter_channels, norm_layer, 'channels_first', 'channels_first'
        )
        
        # 激活函数层：引入非线性
        self.act = build_act_layer(act_layer)
        
        # 第二个卷积层：将中间通道数转换为输出通道数
        # 同样使用3x3卷积核，步长为1，填充为1
        self.conv2 = nn.Conv2d(inter_channels,
                               out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        
        # 第二个归一化层：对输出特征进行归一化
        # 输入输出格式都是channels_first (N, C, H, W)
        self.norm2 = build_norm_layer(
            out_channels, norm_layer, 'channels_first', 'channels_first'
        )

    def forward(self, x):
        """
        前向传播
        
        执行两阶段的特征变换：
        1. 第一阶段：输入 -> 卷积1 -> 归一化1 -> 激活
        2. 第二阶段：激活输出 -> 卷积2 -> 归一化2 -> 输出
        
        Args:
            x: 输入张量，形状为 (N, in_channels, H, W)
            
        Returns:
            输出张量，形状为 (N, out_channels, H, W)
        """
        # 第一阶段：输入通道到中间通道的转换
        x = self.conv1(x)        # 第一个卷积层
        x = self.norm1(x)        # 第一个归一化层
        x = self.act(x)          # 激活函数
        
        # 第二阶段：中间通道到输出通道的转换
        x = self.conv2(x)        # 第二个卷积层
        x = self.norm2(x)        # 第二个归一化层
        
        return x
