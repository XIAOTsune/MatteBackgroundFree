# 导入PyTorch相关模块
import torch
import torch.nn as nn
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
# 导入torchvision预训练模型
from torchvision.models import vgg16, vgg16_bn
from torchvision.models import resnet50

# 导入项目配置和相关模块
from config import Config
from dataset import class_labels_TR_sorted
from models.backbones.build_backbone import build_backbone
from models.modules.decoder_blocks import BasicDecBlk
from models.modules.lateral_blocks import BasicLatBlk
from models.refinement.stem_layer import StemLayer


class RefinerPVTInChannels4(nn.Module):
    """
    基于PVT的4通道输入精细化网络
    
    这个类实现了一个使用PVT (Pyramid Vision Transformer) 作为骨干网络的精细化模型，
    专门处理4通道输入（RGB + 额外通道，如深度或掩码）。
    """
    
    def __init__(self, in_channels=3+1):
        """
        初始化RefinerPVTInChannels4网络
        
        Args:
            in_channels (int): 输入通道数，默认为4（RGB + 1个额外通道）
        """
        super(RefinerPVTInChannels4, self).__init__()
        # 获取配置实例
        self.config = Config()
        # 设置训练轮次
        self.epoch = 1
        # 构建骨干网络，设置输入通道为4
        self.bb = build_backbone(self.config.bb, params_settings='in_channels=4')

        # 定义不同骨干网络的侧向连接通道数配置
        lateral_channels_in_collection = {
            'vgg16': [512, 256, 128, 64], 'vgg16bn': [512, 256, 128, 64], 'resnet50': [1024, 512, 256, 64],
            'pvt_v2_b2': [512, 320, 128, 64], 'pvt_v2_b5': [512, 320, 128, 64],
            'swin_v1_b': [1024, 512, 256, 128], 'swin_v1_l': [1536, 768, 384, 192],
        }
        # 根据配置的骨干网络获取对应的通道数
        channels = lateral_channels_in_collection[self.config.bb]
        # 压缩模块，用于处理最深层特征
        self.squeeze_module = BasicDecBlk(channels[0], channels[0])

        # 解码器模块
        self.decoder = Decoder(channels)

        # 可选：冻结骨干网络参数（当前设置为0，即不冻结）
        if 0:
            for key, value in self.named_parameters():
                if 'bb.' in key:
                    value.requires_grad = False

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量或张量列表
            
        Returns:
            scaled_preds: 多尺度预测结果
        """
        # 如果输入是列表，则在通道维度上拼接
        if isinstance(x, list):
            x = torch.cat(x, dim=1)
            
        ########## 编码器 ##########
        # 根据不同的骨干网络类型进行特征提取
        if self.config.bb in ['vgg16', 'vgg16bn', 'resnet50']:
            # 对于VGG和ResNet，逐层提取特征
            x1 = self.bb.conv1(x)
            x2 = self.bb.conv2(x1)
            x3 = self.bb.conv3(x2)
            x4 = self.bb.conv4(x3)
        else:
            # 对于Transformer类型的骨干网络，直接获取多尺度特征
            x1, x2, x3, x4 = self.bb(x)

        # 对最深层特征进行压缩处理
        x4 = self.squeeze_module(x4)

        ########## 解码器 ##########
        # 将所有特征组织成列表传递给解码器
        features = [x, x1, x2, x3, x4]
        scaled_preds = self.decoder(features)

        return scaled_preds


class Refiner(nn.Module):
    """
    通用精细化网络
    
    这个类实现了一个通用的精细化模型，包含stem层用于预处理输入，
    然后使用指定的骨干网络进行特征提取和解码。
    """
    
    def __init__(self, in_channels=3+1):
        """
        初始化Refiner网络
        
        Args:
            in_channels (int): 输入通道数，默认为4（RGB + 1个额外通道）
        """
        super(Refiner, self).__init__()
        # 获取配置实例
        self.config = Config()
        # 设置训练轮次
        self.epoch = 1
        # Stem层：根据批次大小选择归一化类型
        self.stem_layer = StemLayer(in_channels=in_channels, inter_channels=48, out_channels=3, norm_layer='BN' if self.config.batch_size > 1 else 'LN')
        # 构建骨干网络
        self.bb = build_backbone(self.config.bb)

        # 定义不同骨干网络的侧向连接通道数配置
        lateral_channels_in_collection = {
            'vgg16': [512, 256, 128, 64], 'vgg16bn': [512, 256, 128, 64], 'resnet50': [1024, 512, 256, 64],
            'pvt_v2_b2': [512, 320, 128, 64], 'pvt_v2_b5': [512, 320, 128, 64],
            'swin_v1_b': [1024, 512, 256, 128], 'swin_v1_l': [1536, 768, 384, 192],
        }
        # 根据配置的骨干网络获取对应的通道数
        channels = lateral_channels_in_collection[self.config.bb]
        # 压缩模块，用于处理最深层特征
        self.squeeze_module = BasicDecBlk(channels[0], channels[0])

        # 解码器模块
        self.decoder = Decoder(channels)

        # 可选：冻结骨干网络参数（当前设置为0，即不冻结）
        if 0:
            for key, value in self.named_parameters():
                if 'bb.' in key:
                    value.requires_grad = False

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量或张量列表
            
        Returns:
            scaled_preds: 多尺度预测结果
        """
        # 如果输入是列表，则在通道维度上拼接
        if isinstance(x, list):
            x = torch.cat(x, dim=1)
        # 通过stem层预处理输入
        x = self.stem_layer(x)
        
        ########## 编码器 ##########
        # 根据不同的骨干网络类型进行特征提取
        if self.config.bb in ['vgg16', 'vgg16bn', 'resnet50']:
            # 对于VGG和ResNet，逐层提取特征
            x1 = self.bb.conv1(x)
            x2 = self.bb.conv2(x1)
            x3 = self.bb.conv3(x2)
            x4 = self.bb.conv4(x3)
        else:
            # 对于Transformer类型的骨干网络，直接获取多尺度特征
            x1, x2, x3, x4 = self.bb(x)

        # 对最深层特征进行压缩处理
        x4 = self.squeeze_module(x4)

        ########## 解码器 ##########
        # 将所有特征组织成列表传递给解码器
        features = [x, x1, x2, x3, x4]
        scaled_preds = self.decoder(features)

        return scaled_preds


class Decoder(nn.Module):
    """
    解码器模块
    
    实现特征金字塔网络(FPN)风格的解码器，通过自顶向下的路径和侧向连接
    逐步融合不同尺度的特征，生成最终的预测结果。
    """
    
    def __init__(self, channels):
        """
        初始化解码器
        
        Args:
            channels (list): 各层的通道数列表，从深到浅 [C4, C3, C2, C1]
        """
        super(Decoder, self).__init__()
        # 获取配置实例
        self.config = Config()
        # 动态获取解码块和侧向块类
        DecoderBlock = eval('BasicDecBlk')
        LateralBlock = eval('BasicLatBlk')

        # 构建解码器块：逐步减少通道数
        self.decoder_block4 = DecoderBlock(channels[0], channels[1])  # 最深层解码块
        self.decoder_block3 = DecoderBlock(channels[1], channels[2])  # 第三层解码块
        self.decoder_block2 = DecoderBlock(channels[2], channels[3])  # 第二层解码块
        self.decoder_block1 = DecoderBlock(channels[3], channels[3]//2)  # 第一层解码块

        # 构建侧向连接块：用于融合不同尺度特征
        self.lateral_block4 = LateralBlock(channels[1], channels[1])  # 第四层侧向连接
        self.lateral_block3 = LateralBlock(channels[2], channels[2])  # 第三层侧向连接
        self.lateral_block2 = LateralBlock(channels[3], channels[3])  # 第二层侧向连接

        # 多尺度监督：如果启用，为每个尺度添加输出头
        if self.config.ms_supervision:
            self.conv_ms_spvn_4 = nn.Conv2d(channels[1], 1, 1, 1, 0)  # 第四层监督输出
            self.conv_ms_spvn_3 = nn.Conv2d(channels[2], 1, 1, 1, 0)  # 第三层监督输出
            self.conv_ms_spvn_2 = nn.Conv2d(channels[3], 1, 1, 1, 0)  # 第二层监督输出
        # 最终输出层
        self.conv_out1 = nn.Sequential(nn.Conv2d(channels[3]//2, 1, 1, 1, 0))

    def forward(self, features):
        """
        前向传播
        
        Args:
            features (list): 输入特征列表 [x, x1, x2, x3, x4]，从原图到深层特征
            
        Returns:
            outs (list): 输出预测列表，包含多尺度监督输出（如果启用）和最终输出
        """
        x, x1, x2, x3, x4 = features
        outs = []
        
        # 第四层解码：处理最深层特征
        p4 = self.decoder_block4(x4)
        # 上采样到第三层尺寸并与侧向连接融合
        _p4 = F.interpolate(p4, size=x3.shape[2:], mode='bilinear', align_corners=True)
        _p3 = _p4 + self.lateral_block4(x3)

        # 第三层解码
        p3 = self.decoder_block3(_p3)
        # 上采样到第二层尺寸并与侧向连接融合
        _p3 = F.interpolate(p3, size=x2.shape[2:], mode='bilinear', align_corners=True)
        _p2 = _p3 + self.lateral_block3(x2)

        # 第二层解码
        p2 = self.decoder_block2(_p2)
        # 上采样到第一层尺寸并与侧向连接融合
        _p2 = F.interpolate(p2, size=x1.shape[2:], mode='bilinear', align_corners=True)
        _p1 = _p2 + self.lateral_block2(x1)

        # 第一层解码
        _p1 = self.decoder_block1(_p1)
        # 上采样到原图尺寸
        _p1 = F.interpolate(_p1, size=x.shape[2:], mode='bilinear', align_corners=True)
        # 生成最终输出
        p1_out = self.conv_out1(_p1)

        # 如果启用多尺度监督，添加各层的监督输出
        if self.config.ms_supervision:
            outs.append(self.conv_ms_spvn_4(p4))  # 第四层监督输出
            outs.append(self.conv_ms_spvn_3(p3))  # 第三层监督输出
            outs.append(self.conv_ms_spvn_2(p2))  # 第二层监督输出
        # 添加最终输出
        outs.append(p1_out)
        return outs


class RefUNet(nn.Module):
    """
    基于U-Net的精细化网络
    
    实现了一个简化的U-Net架构用于图像精细化任务。
    包含编码器-解码器结构和跳跃连接。
    """
    
    def __init__(self, in_channels=3+1):
        """
        初始化RefUNet网络
        
        Args:
            in_channels (int): 输入通道数，默认为4（RGB + 1个额外通道）
        """
        super(RefUNet, self).__init__()
        
        # 编码器第一层：两个3x3卷积 + BN + ReLU
        self.encoder_1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1),  # 输入通道到64通道
            nn.Conv2d(64, 64, 3, 1, 1),           # 64通道到64通道
            nn.BatchNorm2d(64),                   # 批归一化
            nn.ReLU(inplace=True)                 # ReLU激活
        )

        # 编码器第二层：最大池化 + 卷积 + BN + ReLU
        self.encoder_2 = nn.Sequential(
            nn.MaxPool2d(2, 2, ceil_mode=True),   # 2x2最大池化，尺寸减半
            nn.Conv2d(64, 64, 3, 1, 1),           # 3x3卷积
            nn.BatchNorm2d(64),                   # 批归一化
            nn.ReLU(inplace=True)                 # ReLU激活
        )

        # 编码器第三层：最大池化 + 卷积 + BN + ReLU
        self.encoder_3 = nn.Sequential(
            nn.MaxPool2d(2, 2, ceil_mode=True),   # 2x2最大池化，尺寸减半
            nn.Conv2d(64, 64, 3, 1, 1),           # 3x3卷积
            nn.BatchNorm2d(64),                   # 批归一化
            nn.ReLU(inplace=True)                 # ReLU激活
        )

        # 编码器第四层：最大池化 + 卷积 + BN + ReLU
        self.encoder_4 = nn.Sequential(
            nn.MaxPool2d(2, 2, ceil_mode=True),   # 2x2最大池化，尺寸减半
            nn.Conv2d(64, 64, 3, 1, 1),           # 3x3卷积
            nn.BatchNorm2d(64),                   # 批归一化
            nn.ReLU(inplace=True)                 # ReLU激活
        )

        # 底部池化层
        self.pool4 = nn.MaxPool2d(2, 2, ceil_mode=True)
        
        #####
        # 解码器第五层（底部）：卷积 + BN + ReLU
        self.decoder_5 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),           # 3x3卷积
            nn.BatchNorm2d(64),                   # 批归一化
            nn.ReLU(inplace=True)                 # ReLU激活
        )
        #####
        
        # 解码器第四层：处理拼接后的128通道特征
        self.decoder_4 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),          # 128通道到64通道
            nn.BatchNorm2d(64),                   # 批归一化
            nn.ReLU(inplace=True)                 # ReLU激活
        )

        # 解码器第三层：处理拼接后的128通道特征
        self.decoder_3 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),          # 128通道到64通道
            nn.BatchNorm2d(64),                   # 批归一化
            nn.ReLU(inplace=True)                 # ReLU激活
        )

        # 解码器第二层：处理拼接后的128通道特征
        self.decoder_2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),          # 128通道到64通道
            nn.BatchNorm2d(64),                   # 批归一化
            nn.ReLU(inplace=True)                 # ReLU激活
        )

        # 解码器第一层：处理拼接后的128通道特征
        self.decoder_1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),          # 128通道到64通道
            nn.BatchNorm2d(64),                   # 批归一化
            nn.ReLU(inplace=True)                 # ReLU激活
        )

        # 最终输出层：64通道到1通道
        self.conv_d0 = nn.Conv2d(64, 1, 3, 1, 1)

        # 2倍上采样层
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量或张量列表
            
        Returns:
            outs (list): 包含最终预测结果的列表
        """
        outs = []
        # 如果输入是列表，则在通道维度上拼接
        if isinstance(x, list):
            x = torch.cat(x, dim=1)
        hx = x

        # 编码器路径：逐层下采样并提取特征
        hx1 = self.encoder_1(hx)    # 第一层编码
        hx2 = self.encoder_2(hx1)   # 第二层编码
        hx3 = self.encoder_3(hx2)   # 第三层编码
        hx4 = self.encoder_4(hx3)   # 第四层编码

        # 底部处理：最深层特征处理
        hx = self.decoder_5(self.pool4(hx4))
        # 上采样并与第四层特征拼接
        hx = torch.cat((self.upscore2(hx), hx4), 1)

        # 解码器路径：逐层上采样并融合跳跃连接
        d4 = self.decoder_4(hx)     # 第四层解码
        # 上采样并与第三层特征拼接
        hx = torch.cat((self.upscore2(d4), hx3), 1)

        d3 = self.decoder_3(hx)     # 第三层解码
        # 上采样并与第二层特征拼接
        hx = torch.cat((self.upscore2(d3), hx2), 1)

        d2 = self.decoder_2(hx)     # 第二层解码
        # 上采样并与第一层特征拼接
        hx = torch.cat((self.upscore2(d2), hx1), 1)

        d1 = self.decoder_1(hx)     # 第一层解码

        # 生成最终输出
        x = self.conv_d0(d1)
        outs.append(x)
        return outs
