# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu, Yutong Lin, Yixuan Wei
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.layers import DropPath, to_2tuple, trunc_normal_

from config import Config


config = Config()

class Mlp(nn.Module):
    """ 多层感知机（Multilayer perceptron）"""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        # 设置输出特征数，默认与输入特征数相同
        out_features = out_features or in_features
        # 设置隐藏层特征数，默认与输入特征数相同
        hidden_features = hidden_features or in_features
        # 第一个全连接层：输入 -> 隐藏层
        self.fc1 = nn.Linear(in_features, hidden_features)
        # 激活函数，默认为GELU
        self.act = act_layer()
        # 第二个全连接层：隐藏层 -> 输出
        self.fc2 = nn.Linear(hidden_features, out_features)
        # Dropout层，用于正则化
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # 前向传播：输入 -> fc1 -> 激活 -> dropout -> fc2 -> dropout -> 输出
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    将输入特征图分割成不重叠的窗口
    Args:
        x: (B, H, W, C) 输入特征图
        window_size (int): 窗口大小

    Returns:
        windows: (num_windows*B, window_size, window_size, C) 分割后的窗口
    """
    B, H, W, C = x.shape
    # 将特征图重塑为窗口形式：(B, H//window_size, window_size, W//window_size, window_size, C)
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # 重新排列维度并合并窗口：(num_windows*B, window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    将窗口重新组合成完整的特征图
    Args:
        windows: (num_windows*B, window_size, window_size, C) 窗口特征
        window_size (int): 窗口大小
        H (int): 图像高度
        W (int): 图像宽度

    Returns:
        x: (B, H, W, C) 重组后的特征图
    """
    C = int(windows.shape[-1])
    # 将窗口重塑为原始形状：(B, H//window_size, W//window_size, window_size, window_size, C)
    x = windows.view(-1, H // window_size, W // window_size, window_size, window_size, C)
    # 重新排列维度并合并：(B, H, W, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    return x


class WindowAttention(nn.Module):
    """ 基于窗口的多头自注意力机制（W-MSA），支持相对位置偏置
    支持移位和非移位窗口

    Args:
        dim (int): 输入通道数
        window_size (tuple[int]): 窗口的高度和宽度
        num_heads (int): 注意力头数
        qkv_bias (bool, optional): 是否为query、key、value添加可学习偏置。默认: True
        qk_scale (float | None, optional): 覆盖默认的qk缩放因子head_dim ** -0.5。默认: None
        attn_drop (float, optional): 注意力权重的dropout比率。默认: 0.0
        proj_drop (float, optional): 输出的dropout比率。默认: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim  # 输入维度
        self.window_size = window_size  # 窗口大小 (Wh, Ww)
        self.num_heads = num_heads  # 注意力头数
        head_dim = dim // num_heads  # 每个头的维度
        self.scale = qk_scale or head_dim ** -0.5  # 缩放因子

        # 定义相对位置偏置参数表
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # 获取窗口内每个token的相对位置索引
        coords_h = torch.arange(self.window_size[0])  # 高度坐标
        coords_w = torch.arange(self.window_size[1])  # 宽度坐标
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        # 计算相对坐标：每个位置相对于其他位置的坐标差
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # 将坐标偏移到从0开始
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        # 注册为缓冲区，不参与梯度更新
        self.register_buffer("relative_position_index", relative_position_index)

        # QKV线性变换层
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop_prob = attn_drop  # 注意力dropout概率
        self.attn_drop = nn.Dropout(attn_drop)  # 注意力dropout层
        # 输出投影层
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # 初始化相对位置偏置表
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)  # Softmax层

    def forward(self, x, mask=None):
        """ 前向传播函数

        Args:
            x: 输入特征，形状为 (num_windows*B, N, C)
            mask: (0/-inf) 掩码，形状为 (num_windows, Wh*Ww, Wh*Ww) 或 None
        """
        B_, N, C = x.shape
        # 计算QKV：(B_, N, 3, num_heads, C//num_heads) -> (3, B_, num_heads, N, C//num_heads)
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # 分离Q、K、V

        # 对Q进行缩放
        q = q * self.scale

        # 如果启用了SDPA（Scaled Dot-Product Attention），使用PyTorch内置的高效实现
        if config.SDPA_enabled:
            x = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None, dropout_p=self.attn_drop_prob, is_causal=False
            ).transpose(1, 2).reshape(B_, N, C)
        else:
            # 手动实现注意力机制
            # 计算注意力分数：Q @ K^T
            attn = (q @ k.transpose(-2, -1))

            # 添加相对位置偏置
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
            )   # Wh*Ww, Wh*Ww, nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

            # 如果有掩码，应用掩码
            if mask is not None:
                nW = mask.shape[0]
                attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
                attn = attn.view(-1, self.num_heads, N, N)
                attn = self.softmax(attn)
            else:
                attn = self.softmax(attn)

            # 应用注意力dropout
            attn = self.attn_drop(attn)

            # 计算输出：注意力权重 @ V
            x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        
        # 输出投影和dropout
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """ Swin Transformer块

    Args:
        dim (int): 输入通道数
        num_heads (int): 注意力头数
        window_size (int): 窗口大小
        shift_size (int): SW-MSA的移位大小
        mlp_ratio (float): MLP隐藏层维度与嵌入维度的比率
        qkv_bias (bool, optional): 是否为query、key、value添加可学习偏置。默认: True
        qk_scale (float | None, optional): 覆盖默认的qk缩放因子head_dim ** -0.5。默认: None
        drop (float, optional): Dropout比率。默认: 0.0
        attn_drop (float, optional): 注意力dropout比率。默认: 0.0
        drop_path (float, optional): 随机深度比率。默认: 0.0
        act_layer (nn.Module, optional): 激活层。默认: nn.GELU
        norm_layer (nn.Module, optional): 归一化层。默认: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim  # 输入维度
        self.num_heads = num_heads  # 注意力头数
        self.window_size = window_size  # 窗口大小
        self.shift_size = shift_size  # 移位大小
        self.mlp_ratio = mlp_ratio  # MLP比率
        # 确保移位大小在有效范围内
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        # 第一个层归一化
        self.norm1 = norm_layer(dim)
        # 窗口注意力模块
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        # 随机深度（Stochastic Depth）
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # 第二个层归一化
        self.norm2 = norm_layer(dim)
        # MLP隐藏层维度
        mlp_hidden_dim = int(dim * mlp_ratio)
        # MLP模块
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # 存储特征图的高度和宽度
        self.H = None
        self.W = None

    def forward(self, x, mask_matrix):
        """ 前向传播函数

        Args:
            x: 输入特征，张量大小 (B, H*W, C)
            H, W: 输入特征的空间分辨率
            mask_matrix: 循环移位的注意力掩码
        """
        B, L, C = x.shape
        H, W = self.H, self.W
        # 确保输入特征大小正确
        assert L == H * W, "input feature has wrong size"

        # 残差连接的输入
        shortcut = x
        # 第一个层归一化
        x = self.norm1(x)
        # 重塑为图像形式：(B, H, W, C)
        x = x.view(B, H, W, C)

        # 将特征图填充到窗口大小的倍数
        pad_l = pad_t = 0  # 左侧和顶部填充为0
        pad_r = (self.window_size - W % self.window_size) % self.window_size  # 右侧填充
        pad_b = (self.window_size - H % self.window_size) % self.window_size  # 底部填充
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # 循环移位
        if self.shift_size > 0:
            # 应用循环移位
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix  # 使用注意力掩码
        else:
            shifted_x = x
            attn_mask = None

        # 分割窗口
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA（窗口多头自注意力/移位窗口多头自注意力）
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # 合并窗口
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # 反向循环移位
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        # 移除填充
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        # 重塑回序列形式：(B, H*W, C)
        x = x.view(B, H * W, C)

        # FFN（前馈网络）
        # 第一个残差连接：输入 + 注意力输出
        x = shortcut + self.drop_path(x)
        # 第二个残差连接：上一步输出 + MLP输出
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Module):
    """ 补丁合并层，用于下采样

    Args:
        dim (int): 输入通道数
        norm_layer (nn.Module, optional): 归一化层。默认: nn.LayerNorm
    """
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        # 线性层：将4*dim维度降到2*dim维度
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        # 归一化层
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """ 前向传播函数

        Args:
            x: 输入特征，张量大小 (B, H*W, C)
            H, W: 输入特征的空间分辨率
        """
        B, L, C = x.shape
        # 确保输入特征大小正确
        assert L == H * W, "input feature has wrong size"

        # 重塑为图像形式：(B, H, W, C)
        x = x.view(B, H, W, C)

        # 如果高度或宽度为奇数，进行填充
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        # 提取四个子采样区域（2x2降采样）
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C - 左上角
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C - 右上角
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C - 左下角
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C - 右下角
        # 在通道维度上连接四个区域
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        # 重塑为序列形式
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        # 归一化和维度降低
        x = self.norm(x)
        x = self.reduction(x)

        return x


class BasicLayer(nn.Module):
    """ Swin Transformer的基本层，对应一个阶段

    Args:
        dim (int): 特征通道数
        depth (int): 该阶段的深度（块数）
        num_heads (int): 注意力头数
        window_size (int): 局部窗口大小。默认: 7
        mlp_ratio (float): MLP隐藏层维度与嵌入维度的比率。默认: 4
        qkv_bias (bool, optional): 是否为query、key、value添加可学习偏置。默认: True
        qk_scale (float | None, optional): 覆盖默认的qk缩放因子head_dim ** -0.5。默认: None
        drop (float, optional): Dropout比率。默认: 0.0
        attn_drop (float, optional): 注意力dropout比率。默认: 0.0
        drop_path (float | tuple[float], optional): 随机深度比率。默认: 0.0
        norm_layer (nn.Module, optional): 归一化层。默认: nn.LayerNorm
        downsample (nn.Module | None, optional): 层末尾的下采样层。默认: None
        use_checkpoint (bool): 是否使用检查点来节省内存。默认: False
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False):
        super().__init__()
        self.window_size = window_size  # 窗口大小
        self.shift_size = window_size // 2  # 移位大小为窗口大小的一半
        self.depth = depth  # 层深度
        self.use_checkpoint = use_checkpoint  # 是否使用检查点

        # 构建Transformer块
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                # 偶数层不移位，奇数层移位
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                # 如果drop_path是列表，使用对应索引的值，否则使用相同值
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # 补丁合并层（下采样）
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        """ 前向传播函数

        Args:
            x: 输入特征，张量大小 (B, H*W, C)
            H, W: 输入特征的空间分辨率
        """

        # 计算SW-MSA的注意力掩码
        # 将int转换为torch.tensor以兼容PyTorch 2.5中的torch.compile
        Hp = torch.ceil(torch.tensor(H) / self.window_size).to(torch.int64) * self.window_size
        Wp = torch.ceil(torch.tensor(W) / self.window_size).to(torch.int64) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        # 定义切片，用于创建掩码
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        # 为不同区域分配不同的标识符
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        # 将掩码分割成窗口
        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        # 创建注意力掩码：相同区域内的token可以互相注意，不同区域的token被掩码
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0)).to(x.dtype)

        # 通过所有Transformer块
        for blk in self.blocks:
            blk.H, blk.W = H, W  # 设置块的空间分辨率
            if self.use_checkpoint:
                # 使用检查点来节省内存
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)
        
        # 如果有下采样层，应用下采样
        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2  # 下采样后的分辨率
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W


class PatchEmbed(nn.Module):
    """ 图像到补丁嵌入

    Args:
        patch_size (int): 补丁token大小。默认: 4
        in_channels (int): 输入图像通道数。默认: 3
        embed_dim (int): 线性投影输出通道数。默认: 96
        norm_layer (nn.Module, optional): 归一化层。默认: None
    """

    def __init__(self, patch_size=4, in_channels=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)  # 转换为元组形式
        self.patch_size = patch_size

        self.in_channels = in_channels  # 输入通道数
        self.embed_dim = embed_dim  # 嵌入维度

        # 使用卷积进行补丁嵌入：将图像分割成补丁并投影到嵌入空间
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        # 可选的归一化层
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """前向传播函数"""
        # 获取输入尺寸
        _, _, H, W = x.size()
        # 如果宽度不能被补丁大小整除，进行填充
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        # 如果高度不能被补丁大小整除，进行填充
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        # 应用卷积投影：(B, C, H, W) -> (B, embed_dim, Wh, Ww)
        x = self.proj(x)  # B C Wh Ww
        # 如果有归一化层，应用归一化
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            # 展平并转置：(B, embed_dim, Wh, Ww) -> (B, Wh*Ww, embed_dim)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            # 转置并重塑回原形状：(B, Wh*Ww, embed_dim) -> (B, embed_dim, Wh, Ww)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)

        return x


class SwinTransformer(nn.Module):
    """ Swin Transformer骨干网络
        PyTorch实现：`Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        pretrain_img_size (int): 预训练模型的输入图像大小，用于绝对位置嵌入。默认224
        patch_size (int | tuple(int)): 补丁大小。默认: 4
        in_channels (int): 输入图像通道数。默认: 3
        embed_dim (int): 线性投影输出通道数。默认: 96
        depths (tuple[int]): 每个Swin Transformer阶段的深度
        num_heads (tuple[int]): 每个阶段的注意力头数
        window_size (int): 窗口大小。默认: 7
        mlp_ratio (float): MLP隐藏层维度与嵌入维度的比率。默认: 4
        qkv_bias (bool): 是否为query、key、value添加可学习偏置。默认: True
        qk_scale (float): 覆盖默认的qk缩放因子head_dim ** -0.5
        drop_rate (float): Dropout比率
        attn_drop_rate (float): 注意力dropout比率。默认: 0
        drop_path_rate (float): 随机深度比率。默认: 0.2
        norm_layer (nn.Module): 归一化层。默认: nn.LayerNorm
        ape (bool): 是否为补丁嵌入添加绝对位置嵌入。默认: False
        patch_norm (bool): 是否在补丁嵌入后添加归一化。默认: True
        out_indices (Sequence[int]): 输出哪些阶段的特征
        frozen_stages (int): 冻结的阶段数（停止梯度并设置为评估模式）
            -1表示不冻结任何参数
        use_checkpoint (bool): 是否使用检查点来节省内存。默认: False
    """

    def __init__(self,
                 pretrain_img_size=224,
                 patch_size=4,
                 in_channels=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 use_checkpoint=False):
        super().__init__()

        self.pretrain_img_size = pretrain_img_size  # 预训练图像大小
        self.num_layers = len(depths)  # 层数
        self.embed_dim = embed_dim  # 嵌入维度
        self.ape = ape  # 是否使用绝对位置嵌入
        self.patch_norm = patch_norm  # 是否使用补丁归一化
        self.out_indices = out_indices  # 输出索引
        self.frozen_stages = frozen_stages  # 冻结阶段

        # 将图像分割成非重叠的补丁
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # 绝对位置嵌入
        if self.ape:
            pretrain_img_size = to_2tuple(pretrain_img_size)
            patch_size = to_2tuple(patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1]]

            # 创建绝对位置嵌入参数
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1]))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        # 位置dropout
        self.pos_drop = nn.Dropout(p=drop_rate)

        # 随机深度衰减规则
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # 构建各层
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),  # 每层维度翻倍
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                # 为每层分配对应的随机深度比率
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                # 除了最后一层，其他层都有下采样
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        # 计算每层的特征数
        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        # 为每个输出层添加归一化层
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        # 冻结指定阶段
        self._freeze_stages()

    def _freeze_stages(self):
        """冻结指定阶段的参数"""
        if self.frozen_stages >= 0:
            # 冻结补丁嵌入层
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            # 冻结绝对位置嵌入
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            # 冻结位置dropout和指定的Transformer层
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False


    def forward(self, x):
        """前向传播函数"""
        # 补丁嵌入
        x = self.patch_embed(x)

        Wh, Ww = x.size(2), x.size(3)  # 获取补丁分辨率
        if self.ape:
            # 将位置嵌入插值到对应大小
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode='bicubic')
            x = (x + absolute_pos_embed) # B Wh*Ww C
            
        outs = []  # 存储输出特征
        # 展平并转置：(B, C, Wh, Ww) -> (B, Wh*Ww, C)
        x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)
        
        # 通过所有层
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)

            # 如果当前层在输出索引中，添加到输出列表
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)

                # 重塑为图像格式：(B, H*W, C) -> (B, C, H, W)
                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        return tuple(outs)

    def train(self, mode=True):
        """将模型转换为训练模式，同时保持冻结层的状态"""
        super(SwinTransformer, self).train(mode)
        self._freeze_stages()

def swin_v1_t():
    """Swin Transformer Tiny模型"""
    model = SwinTransformer(embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=7)
    return model

def swin_v1_s():
    """Swin Transformer Small模型"""
    model = SwinTransformer(embed_dim=96, depths=[2, 2, 18, 2], num_heads=[3, 6, 12, 24], window_size=7)
    return model

def swin_v1_b():
    """Swin Transformer Base模型"""
    model = SwinTransformer(embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32], window_size=12)
    return model

def swin_v1_l():
    """Swin Transformer Large模型"""
    model = SwinTransformer(embed_dim=192, depths=[2, 2, 18, 2], num_heads=[6, 12, 24, 48], window_size=12)
    return model
