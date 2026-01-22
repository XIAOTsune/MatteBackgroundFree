# 导入PyTorch神经网络模块
import torch.nn as nn


def build_act_layer(act_layer):
    """
    构建激活函数层
    
    根据指定的激活函数名称创建对应的激活函数层。
    支持常用的激活函数：ReLU、SiLU、GELU。
    
    Args:
        act_layer (str): 激活函数名称，支持 'ReLU', 'SiLU', 'GELU'
        
    Returns:
        nn.Module: 对应的激活函数层
        
    Raises:
        NotImplementedError: 当指定的激活函数不被支持时抛出异常
    """
    if act_layer == 'ReLU':
        # ReLU激活函数，使用原地操作节省内存
        return nn.ReLU(inplace=True)
    elif act_layer == 'SiLU':
        # SiLU (Sigmoid Linear Unit) 激活函数，也称为Swish
        return nn.SiLU(inplace=True)
    elif act_layer == 'GELU':
        # GELU (Gaussian Error Linear Unit) 激活函数
        return nn.GELU()

    # 如果指定的激活函数不被支持，抛出异常
    raise NotImplementedError(f'build_act_layer does not support {act_layer}')


def build_norm_layer(dim,
                     norm_layer,
                     in_format='channels_last',
                     out_format='channels_last',
                     eps=1e-6):
    """
    构建归一化层
    
    根据指定的归一化类型和输入输出格式创建归一化层。
    支持BatchNorm2d和LayerNorm，并能够处理不同的数据格式转换。
    
    Args:
        dim (int): 归一化的维度大小
        norm_layer (str): 归一化类型，支持 'BN' (BatchNorm) 和 'LN' (LayerNorm)
        in_format (str): 输入数据格式，'channels_first' 或 'channels_last'，默认 'channels_last'
        out_format (str): 输出数据格式，'channels_first' 或 'channels_last'，默认 'channels_last'
        eps (float): LayerNorm的epsilon参数，默认1e-6
        
    Returns:
        nn.Sequential: 包含归一化层和必要格式转换层的序列模块
        
    Raises:
        NotImplementedError: 当指定的归一化类型不被支持时抛出异常
    """
    layers = []
    
    if norm_layer == 'BN':
        # BatchNorm2d需要channels_first格式 (N, C, H, W)
        if in_format == 'channels_last':
            # 如果输入是channels_last格式，先转换为channels_first
            layers.append(to_channels_first())
        # 添加BatchNorm2d层
        layers.append(nn.BatchNorm2d(dim))
        if out_format == 'channels_last':
            # 如果需要输出channels_last格式，转换回去
            layers.append(to_channels_last())
            
    elif norm_layer == 'LN':
        # LayerNorm需要channels_last格式 (N, H, W, C)
        if in_format == 'channels_first':
            # 如果输入是channels_first格式，先转换为channels_last
            layers.append(to_channels_last())
        # 添加LayerNorm层
        layers.append(nn.LayerNorm(dim, eps=eps))
        if out_format == 'channels_first':
            # 如果需要输出channels_first格式，转换回去
            layers.append(to_channels_first())
    else:
        # 如果指定的归一化类型不被支持，抛出异常
        raise NotImplementedError(
            f'build_norm_layer does not support {norm_layer}')
    
    # 返回包含所有层的序列模块
    return nn.Sequential(*layers)


class to_channels_first(nn.Module):
    """
    数据格式转换：从channels_last转换为channels_first
    
    将输入张量从 (N, H, W, C) 格式转换为 (N, C, H, W) 格式。
    这种转换通常用于从LayerNorm兼容格式转换为Conv2d兼容格式。
    """

    def __init__(self):
        """初始化格式转换模块"""
        super().__init__()

    def forward(self, x):
        """
        前向传播：执行格式转换
        
        Args:
            x: 输入张量，形状为 (N, H, W, C)
            
        Returns:
            转换后的张量，形状为 (N, C, H, W)
        """
        # 使用permute重新排列维度：(0, 3, 1, 2) 对应 (N, C, H, W)
        return x.permute(0, 3, 1, 2)


class to_channels_last(nn.Module):
    """
    数据格式转换：从channels_first转换为channels_last
    
    将输入张量从 (N, C, H, W) 格式转换为 (N, H, W, C) 格式。
    这种转换通常用于从Conv2d兼容格式转换为LayerNorm兼容格式。
    """

    def __init__(self):
        """初始化格式转换模块"""
        super().__init__()

    def forward(self, x):
        """
        前向传播：执行格式转换
        
        Args:
            x: 输入张量，形状为 (N, C, H, W)
            
        Returns:
            转换后的张量，形状为 (N, H, W, C)
        """
        # 使用permute重新排列维度：(0, 2, 3, 1) 对应 (N, H, W, C)
        return x.permute(0, 2, 3, 1)
