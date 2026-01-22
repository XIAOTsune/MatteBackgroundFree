# 导入PyTorch相关模块
import torch
import torch.nn as nn
# 导入ASPP模块
from models.modules.aspp import ASPP, ASPPDeformable
# 导入配置文件
from config import Config

# 创建配置实例
config = Config()


class BasicDecBlk(nn.Module):
    """
    基础解码器块
    用于特征解码和上采样，包含卷积、批归一化、激活函数和可选的注意力机制
    """
    def __init__(self, in_channels=64, out_channels=64, inter_channels=64):
        """
        初始化基础解码器块
        
        Args:
            in_channels: 输入通道数，默认64
            out_channels: 输出通道数，默认64
            inter_channels: 中间层通道数，默认64（实际会根据配置调整）
        """
        super(BasicDecBlk, self).__init__()
        # 根据配置设置中间层通道数：自适应模式下为输入通道数的1/4，否则为64
        inter_channels = in_channels // 4 if config.dec_channels_inter == 'adap' else 64
        # 输入卷积层，3x3卷积，步长1，填充1
        self.conv_in = nn.Conv2d(in_channels, inter_channels, 3, 1, padding=1)
        # ReLU激活函数，使用原地操作节省内存
        self.relu_in = nn.ReLU(inplace=True)
        
        # 根据配置选择注意力机制
        if config.dec_att == 'ASPP':
            # 使用标准ASPP注意力机制
            self.dec_att = ASPP(in_channels=inter_channels)
        elif config.dec_att == 'ASPPDeformable':
            # 使用可变形ASPP注意力机制
            self.dec_att = ASPPDeformable(in_channels=inter_channels)
        
        # 输出卷积层，3x3卷积，步长1，填充1
        self.conv_out = nn.Conv2d(inter_channels, out_channels, 3, 1, padding=1)
        # 输入批归一化层，如果批大小大于1则使用BatchNorm，否则使用Identity
        self.bn_in = nn.BatchNorm2d(inter_channels) if config.batch_size > 1 else nn.Identity()
        # 输出批归一化层
        self.bn_out = nn.BatchNorm2d(out_channels) if config.batch_size > 1 else nn.Identity()

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入特征图
            
        Returns:
            经过解码处理的特征图
        """
        # 输入卷积
        x = self.conv_in(x)
        # 输入批归一化
        x = self.bn_in(x)
        # ReLU激活
        x = self.relu_in(x)
        # 如果存在注意力机制，则应用注意力
        if hasattr(self, 'dec_att'):
            x = self.dec_att(x)
        # 输出卷积
        x = self.conv_out(x)
        # 输出批归一化
        x = self.bn_out(x)
        return x


class ResBlk(nn.Module):
    """
    残差解码器块
    在基础解码器块的基础上添加残差连接，有助于梯度传播和特征保持
    """
    def __init__(self, in_channels=64, out_channels=None, inter_channels=64):
        """
        初始化残差解码器块
        
        Args:
            in_channels: 输入通道数，默认64
            out_channels: 输出通道数，如果为None则等于输入通道数
            inter_channels: 中间层通道数，默认64（实际会根据配置调整）
        """
        super(ResBlk, self).__init__()
        # 如果未指定输出通道数，则设为输入通道数
        if out_channels is None:
            out_channels = in_channels
        # 根据配置设置中间层通道数：自适应模式下为输入通道数的1/4，否则为64
        inter_channels = in_channels // 4 if config.dec_channels_inter == 'adap' else 64

        # 输入卷积层，3x3卷积，步长1，填充1
        self.conv_in = nn.Conv2d(in_channels, inter_channels, 3, 1, padding=1)
        # 输入批归一化层
        self.bn_in = nn.BatchNorm2d(inter_channels) if config.batch_size > 1 else nn.Identity()
        # ReLU激活函数
        self.relu_in = nn.ReLU(inplace=True)

        # 根据配置选择注意力机制
        if config.dec_att == 'ASPP':
            # 使用标准ASPP注意力机制
            self.dec_att = ASPP(in_channels=inter_channels)
        elif config.dec_att == 'ASPPDeformable':
            # 使用可变形ASPP注意力机制
            self.dec_att = ASPPDeformable(in_channels=inter_channels)

        # 输出卷积层，3x3卷积，步长1，填充1
        self.conv_out = nn.Conv2d(inter_channels, out_channels, 3, 1, padding=1)
        # 输出批归一化层
        self.bn_out = nn.BatchNorm2d(out_channels) if config.batch_size > 1 else nn.Identity()
        
        # 残差连接的1x1卷积，用于调整通道数匹配
        self.conv_resi = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入特征图
            
        Returns:
            经过残差解码处理的特征图
        """
        # 保存残差连接的输入，通过1x1卷积调整通道数
        _x = self.conv_resi(x)
        # 主分支：输入卷积
        x = self.conv_in(x)
        # 输入批归一化
        x = self.bn_in(x)
        # ReLU激活
        x = self.relu_in(x)
        # 如果存在注意力机制，则应用注意力
        if hasattr(self, 'dec_att'):
            x = self.dec_att(x)
        # 输出卷积
        x = self.conv_out(x)
        # 输出批归一化
        x = self.bn_out(x)
        # 残差连接：主分支输出 + 残差分支输出
        return x + _x
