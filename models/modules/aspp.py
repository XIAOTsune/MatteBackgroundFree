# 导入PyTorch相关模块
import torch
import torch.nn as nn
import torch.nn.functional as F
# 导入可变形卷积模块
from models.modules.deform_conv import DeformableConv2d
# 导入配置文件
from config import Config

# 创建配置实例
config = Config()


class _ASPPModule(nn.Module):
    """
    ASPP（Atrous Spatial Pyramid Pooling）基础模块
    使用不同膨胀率的空洞卷积来捕获多尺度上下文信息
    """
    def __init__(self, in_channels, planes, kernel_size, padding, dilation):
        """
        初始化ASPP基础模块
        
        Args:
            in_channels: 输入通道数
            planes: 输出通道数
            kernel_size: 卷积核大小
            padding: 填充大小
            dilation: 膨胀率
        """
        super(_ASPPModule, self).__init__()
        # 空洞卷积层，使用指定的膨胀率
        self.atrous_conv = nn.Conv2d(in_channels, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        # 批归一化层，如果批大小大于1则使用BatchNorm，否则使用Identity
        self.bn = nn.BatchNorm2d(planes) if config.batch_size > 1 else nn.Identity()
        # ReLU激活函数，使用原地操作节省内存
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入特征图
            
        Returns:
            经过空洞卷积、批归一化和ReLU激活的特征图
        """
        # 应用空洞卷积
        x = self.atrous_conv(x)
        # 应用批归一化
        x = self.bn(x)
        # 应用ReLU激活并返回
        return self.relu(x)


class ASPP(nn.Module):
    """
    ASPP（Atrous Spatial Pyramid Pooling）模块
    通过并行的多个不同膨胀率的空洞卷积和全局平均池化来捕获多尺度上下文信息
    """
    def __init__(self, in_channels=64, out_channels=None, output_stride=16):
        """
        初始化ASPP模块
        
        Args:
            in_channels: 输入通道数，默认64
            out_channels: 输出通道数，如果为None则等于输入通道数
            output_stride: 输出步长，支持16和8，决定膨胀率的设置
        """
        super(ASPP, self).__init__()
        # 下采样比例
        self.down_scale = 1
        # 如果未指定输出通道数，则设为输入通道数
        if out_channels is None:
            out_channels = in_channels
        # 中间层通道数
        self.in_channelster = 256 // self.down_scale
        
        # 根据输出步长设置不同的膨胀率
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        # 创建四个不同膨胀率的ASPP模块
        # 1x1卷积，膨胀率为1
        self.aspp1 = _ASPPModule(in_channels, self.in_channelster, 1, padding=0, dilation=dilations[0])
        # 3x3卷积，膨胀率为6/12
        self.aspp2 = _ASPPModule(in_channels, self.in_channelster, 3, padding=dilations[1], dilation=dilations[1])
        # 3x3卷积，膨胀率为12/24
        self.aspp3 = _ASPPModule(in_channels, self.in_channelster, 3, padding=dilations[2], dilation=dilations[2])
        # 3x3卷积，膨胀率为18/36
        self.aspp4 = _ASPPModule(in_channels, self.in_channelster, 3, padding=dilations[3], dilation=dilations[3])

        # 全局平均池化分支
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # 自适应全局平均池化到1x1
            nn.Conv2d(in_channels, self.in_channelster, 1, stride=1, bias=False),  # 1x1卷积
            nn.BatchNorm2d(self.in_channelster) if config.batch_size > 1 else nn.Identity(),  # 批归一化
            nn.ReLU(inplace=True)  # ReLU激活
        )
        # 融合所有分支的1x1卷积
        self.conv1 = nn.Conv2d(self.in_channelster * 5, out_channels, 1, bias=False)
        # 最终的批归一化层
        self.bn1 = nn.BatchNorm2d(out_channels) if config.batch_size > 1 else nn.Identity()
        # ReLU激活函数
        self.relu = nn.ReLU(inplace=True)
        # Dropout层，防止过拟合
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入特征图
            
        Returns:
            融合多尺度上下文信息后的特征图
        """
        # 通过四个不同膨胀率的ASPP模块
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        # 全局平均池化分支
        x5 = self.global_avg_pool(x)
        # 将全局特征上采样到与其他分支相同的尺寸
        x5 = F.interpolate(x5, size=x1.size()[2:], mode='bilinear', align_corners=True)
        # 在通道维度上连接所有分支
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        # 通过1x1卷积融合特征
        x = self.conv1(x)
        # 应用批归一化
        x = self.bn1(x)
        # 应用ReLU激活
        x = self.relu(x)

        # 应用Dropout并返回
        return self.dropout(x)


##################### Deformable
class _ASPPModuleDeformable(nn.Module):
    """
    可变形ASPP基础模块
    使用可变形卷积替代标准卷积，能够自适应地调整感受野形状
    """
    def __init__(self, in_channels, planes, kernel_size, padding):
        """
        初始化可变形ASPP基础模块
        
        Args:
            in_channels: 输入通道数
            planes: 输出通道数
            kernel_size: 卷积核大小
            padding: 填充大小
        """
        super(_ASPPModuleDeformable, self).__init__()
        # 可变形卷积层
        self.atrous_conv = DeformableConv2d(in_channels, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, bias=False)
        # 批归一化层
        self.bn = nn.BatchNorm2d(planes) if config.batch_size > 1 else nn.Identity()
        # ReLU激活函数
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入特征图
            
        Returns:
            经过可变形卷积、批归一化和ReLU激活的特征图
        """
        # 应用可变形卷积
        x = self.atrous_conv(x)
        # 应用批归一化
        x = self.bn(x)
        # 应用ReLU激活并返回
        return self.relu(x)


class ASPPDeformable(nn.Module):
    """
    可变形ASPP模块
    使用可变形卷积构建的ASPP，能够更灵活地捕获多尺度上下文信息
    """
    def __init__(self, in_channels, out_channels=None, parallel_block_sizes=[1, 3, 7]):
        """
        初始化可变形ASPP模块
        
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数，如果为None则等于输入通道数
            parallel_block_sizes: 并行卷积块的核大小列表，默认[1, 3, 7]
        """
        super(ASPPDeformable, self).__init__()
        # 下采样比例
        self.down_scale = 1
        # 如果未指定输出通道数，则设为输入通道数
        if out_channels is None:
            out_channels = in_channels
        # 中间层通道数
        self.in_channelster = 256 // self.down_scale

        # 1x1卷积分支
        self.aspp1 = _ASPPModuleDeformable(in_channels, self.in_channelster, 1, padding=0)
        # 创建多个不同核大小的可变形卷积分支
        self.aspp_deforms = nn.ModuleList([
            _ASPPModuleDeformable(in_channels, self.in_channelster, conv_size, padding=int(conv_size//2)) 
            for conv_size in parallel_block_sizes
        ])

        # 全局平均池化分支
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # 自适应全局平均池化到1x1
            nn.Conv2d(in_channels, self.in_channelster, 1, stride=1, bias=False),  # 1x1卷积
            nn.BatchNorm2d(self.in_channelster) if config.batch_size > 1 else nn.Identity(),  # 批归一化
            nn.ReLU(inplace=True)  # ReLU激活
        )
        # 融合所有分支的1x1卷积，通道数为：1x1分支 + 可变形分支数量 + 全局池化分支
        self.conv1 = nn.Conv2d(self.in_channelster * (2 + len(self.aspp_deforms)), out_channels, 1, bias=False)
        # 最终的批归一化层
        self.bn1 = nn.BatchNorm2d(out_channels) if config.batch_size > 1 else nn.Identity()
        # ReLU激活函数
        self.relu = nn.ReLU(inplace=True)
        # Dropout层，防止过拟合
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入特征图
            
        Returns:
            融合多尺度上下文信息后的特征图
        """
        # 1x1卷积分支
        x1 = self.aspp1(x)
        # 多个可变形卷积分支
        x_aspp_deforms = [aspp_deform(x) for aspp_deform in self.aspp_deforms]
        # 全局平均池化分支
        x5 = self.global_avg_pool(x)
        # 将全局特征上采样到与其他分支相同的尺寸
        x5 = F.interpolate(x5, size=x1.size()[2:], mode='bilinear', align_corners=True)
        # 在通道维度上连接所有分支
        x = torch.cat((x1, *x_aspp_deforms, x5), dim=1)

        # 通过1x1卷积融合特征
        x = self.conv1(x)
        # 应用批归一化
        x = self.bn1(x)
        # 应用ReLU激活
        x = self.relu(x)

        # 应用Dropout并返回
        return self.dropout(x)
