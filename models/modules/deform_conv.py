# 导入PyTorch相关模块
import torch
import torch.nn as nn
# 导入torchvision中的可变形卷积操作
from torchvision.ops import deform_conv2d


class DeformableConv2d(nn.Module):
    """
    可变形卷积2D模块
    
    可变形卷积通过学习偏移量来调整卷积核的采样位置，使得卷积操作能够适应不同的几何变换。
    相比标准卷积，可变形卷积具有更强的几何建模能力，特别适用于处理具有复杂几何结构的对象。
    
    该实现包含三个主要组件：
    1. 偏移量预测网络：预测每个卷积核位置的2D偏移量
    2. 调制因子预测网络：预测每个位置的重要性权重
    3. 标准卷积核：执行实际的卷积操作
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=False):
        """
        初始化可变形卷积模块
        
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_size: 卷积核大小，可以是int或tuple，默认3
            stride: 步长，可以是int或tuple，默认1
            padding: 填充大小，默认1
            bias: 是否使用偏置，默认False
        """
        super(DeformableConv2d, self).__init__()
        
        # 确保kernel_size是tuple或int类型
        assert type(kernel_size) == tuple or type(kernel_size) == int

        # 将kernel_size和stride转换为tuple格式
        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        
        # 偏移量预测卷积层
        # 输出通道数为 2 * kernel_size[0] * kernel_size[1]，因为每个卷积核位置需要预测x和y两个方向的偏移量
        self.offset_conv = nn.Conv2d(in_channels,
                                     2 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=self.padding,
                                     bias=True)

        # 将偏移量卷积的权重和偏置初始化为0，确保初始时没有偏移
        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)
        
        # 调制因子预测卷积层
        # 输出通道数为 1 * kernel_size[0] * kernel_size[1]，为每个卷积核位置预测一个调制因子
        self.modulator_conv = nn.Conv2d(in_channels,
                                     1 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=self.padding,
                                     bias=True)

        # 将调制因子卷积的权重和偏置初始化为0
        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)

        # 标准卷积层，执行实际的卷积操作
        self.regular_conv = nn.Conv2d(in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      bias=bias)

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入特征图，形状为 (N, C, H, W)
            
        Returns:
            经过可变形卷积处理的特征图
        """
        # 可选的偏移量限制（当前被注释掉）
        # 这可以用来限制偏移量的最大值，防止偏移过大
        #h, w = x.shape[2:]
        #max_offset = max(h, w)/4.

        # 预测偏移量，形状为 (N, 2*K*K, H, W)，其中K是卷积核大小
        offset = self.offset_conv(x)#.clamp(-max_offset, max_offset)
        
        # 预测调制因子，通过sigmoid激活函数确保值在(0,2)范围内
        # 乘以2是为了让调制因子的期望值为1，形状为 (N, K*K, H, W)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))
        
        # 执行可变形卷积操作
        x = deform_conv2d(
            input=x,                        # 输入特征图
            offset=offset,                  # 偏移量
            weight=self.regular_conv.weight, # 卷积权重
            bias=self.regular_conv.bias,    # 卷积偏置
            padding=self.padding,           # 填充
            mask=modulator,                 # 调制因子（掩码）
            stride=self.stride,             # 步长
        )
        return x
