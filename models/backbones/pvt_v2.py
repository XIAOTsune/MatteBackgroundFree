# 导入数学库，用于数学计算
import math
# 导入 functools 中的 partial 函数，用于创建偏函数
from functools import partial
# 导入 PyTorch 核心库
import torch
# 导入 PyTorch 神经网络模块
import torch.nn as nn

# 从 timm 库导入 DropPath（随机深度）、to_2tuple（转换为二元组）、trunc_normal_（截断正态分布初始化）
from timm.layers import DropPath, to_2tuple, trunc_normal_

# 导入配置文件
from config import Config

# 创建配置实例
config = Config()


class Mlp(nn.Module):
    """
    多层感知机（MLP）模块，用于 Transformer 中的前馈网络
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        """
        初始化 MLP 模块
        
        Args:
            in_features (int): 输入特征维度
            hidden_features (int): 隐藏层特征维度，默认为输入特征维度
            out_features (int): 输出特征维度，默认为输入特征维度
            act_layer: 激活函数层，默认为 GELU
            drop (float): Dropout 概率
        """
        super().__init__()
        # 如果未指定输出特征维度，则使用输入特征维度
        out_features = out_features or in_features
        # 如果未指定隐藏层特征维度，则使用输入特征维度
        hidden_features = hidden_features or in_features
        # 第一个全连接层，将输入特征映射到隐藏层
        self.fc1 = nn.Linear(in_features, hidden_features)
        # 深度可分离卷积层，用于增强特征表示
        self.dwconv = DWConv(hidden_features)
        # 激活函数
        self.act = act_layer()
        # 第二个全连接层，将隐藏层特征映射到输出
        self.fc2 = nn.Linear(hidden_features, out_features)
        # Dropout 层，用于正则化
        self.drop = nn.Dropout(drop)

        # 应用权重初始化
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        权重初始化函数
        
        Args:
            m: 网络模块
        """
        # 如果是线性层
        if isinstance(m, nn.Linear):
            # 使用截断正态分布初始化权重
            trunc_normal_(m.weight, std=.02)
            # 如果有偏置项，初始化为 0
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        # 如果是 LayerNorm 层
        elif isinstance(m, nn.LayerNorm):
            # 偏置项初始化为 0
            nn.init.constant_(m.bias, 0)
            # 权重初始化为 1.0
            nn.init.constant_(m.weight, 1.0)
        # 如果是卷积层
        elif isinstance(m, nn.Conv2d):
            # 计算 fan_out（输出连接数）
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            # 使用正态分布初始化权重
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            # 如果有偏置项，初始化为 0
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        """
        前向传播函数
        
        Args:
            x: 输入特征张量 [B, N, C]
            H: 特征图高度
            W: 特征图宽度
        
        Returns:
            输出特征张量
        """
        # 第一个全连接层
        x = self.fc1(x)
        # 深度可分离卷积
        x = self.dwconv(x, H, W)
        # 激活函数
        x = self.act(x)
        # Dropout
        x = self.drop(x)
        # 第二个全连接层
        x = self.fc2(x)
        # Dropout
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """
    空间缩减注意力（Spatial-Reduction Attention）模块
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        """
        初始化注意力模块
        
        Args:
            dim (int): 输入特征维度
            num_heads (int): 注意力头数
            qkv_bias (bool): 是否在 QKV 线性层中使用偏置
            qk_scale (float): QK 缩放因子，默认为 head_dim ** -0.5
            attn_drop (float): 注意力 Dropout 概率
            proj_drop (float): 投影 Dropout 概率
            sr_ratio (int): 空间缩减比例
        """
        super().__init__()
        # 确保特征维度能被注意力头数整除
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        # 保存参数
        self.dim = dim
        self.num_heads = num_heads
        # 计算每个注意力头的维度
        head_dim = dim // num_heads
        # 设置缩放因子
        self.scale = qk_scale or head_dim ** -0.5

        # Query 线性变换
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        # Key 和 Value 线性变换（合并为一个）
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        # 注意力 Dropout 概率（用于 SDPA）
        self.attn_drop_prob = attn_drop
        # 注意力 Dropout 层
        self.attn_drop = nn.Dropout(attn_drop)
        # 输出投影层
        self.proj = nn.Linear(dim, dim)
        # 投影 Dropout 层
        self.proj_drop = nn.Dropout(proj_drop)

        # 空间缩减比例
        self.sr_ratio = sr_ratio
        # 如果空间缩减比例大于 1，创建空间缩减层
        if sr_ratio > 1:
            # 空间缩减卷积层
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            # LayerNorm 层
            self.norm = nn.LayerNorm(dim)

        # 应用权重初始化
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        权重初始化函数
        
        Args:
            m: 网络模块
        """
        # 如果是线性层
        if isinstance(m, nn.Linear):
            # 使用截断正态分布初始化权重
            trunc_normal_(m.weight, std=.02)
            # 如果有偏置项，初始化为 0
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        # 如果是 LayerNorm 层
        elif isinstance(m, nn.LayerNorm):
            # 偏置项初始化为 0
            nn.init.constant_(m.bias, 0)
            # 权重初始化为 1.0
            nn.init.constant_(m.weight, 1.0)
        # 如果是卷积层
        elif isinstance(m, nn.Conv2d):
            # 计算 fan_out（输出连接数）
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            # 使用正态分布初始化权重
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            # 如果有偏置项，初始化为 0
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        """
        前向传播函数
        
        Args:
            x: 输入特征张量 [B, N, C]
            H: 特征图高度
            W: 特征图宽度
        
        Returns:
            输出特征张量
        """
        # 获取输入张量的形状
        B, N, C = x.shape
        # 计算 Query，并重塑为多头注意力格式
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # 如果使用空间缩减
        if self.sr_ratio > 1:
            # 将输入重塑为图像格式
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            # 应用空间缩减卷积
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            # 应用 LayerNorm
            x_ = self.norm(x_)
            # 计算 Key 和 Value
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            # 直接计算 Key 和 Value
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # 分离 Key 和 Value
        k, v = kv[0], kv[1]

        # 如果启用了 SDPA（Scaled Dot-Product Attention）
        if config.SDPA_enabled:
            # 使用 PyTorch 的优化注意力实现
            x = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None, dropout_p=self.attn_drop_prob, is_causal=False
            ).transpose(1, 2).reshape(B, N, C)
        else:
            # 手动实现注意力计算
            # 计算注意力分数
            attn = (q @ k.transpose(-2, -1)) * self.scale
            # 应用 Softmax
            attn = attn.softmax(dim=-1)
            # 应用注意力 Dropout
            attn = self.attn_drop(attn)

            # 计算注意力输出
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # 应用输出投影
        x = self.proj(x)
        # 应用投影 Dropout
        x = self.proj_drop(x)

        return x


class Block(nn.Module):
    """
    Transformer 块，包含注意力和 MLP 模块
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        """
        初始化 Transformer 块
        
        Args:
            dim (int): 输入特征维度
            num_heads (int): 注意力头数
            mlp_ratio (float): MLP 隐藏层维度与输入维度的比例
            qkv_bias (bool): 是否在 QKV 线性层中使用偏置
            qk_scale (float): QK 缩放因子
            drop (float): Dropout 概率
            attn_drop (float): 注意力 Dropout 概率
            drop_path (float): DropPath 概率
            act_layer: 激活函数层
            norm_layer: 归一化层
            sr_ratio (int): 空间缩减比例
        """
        super().__init__()
        # 第一个 LayerNorm 层
        self.norm1 = norm_layer(dim)
        # 注意力模块
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # DropPath 层，用于随机深度正则化
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # 第二个 LayerNorm 层
        self.norm2 = norm_layer(dim)
        # 计算 MLP 隐藏层维度
        mlp_hidden_dim = int(dim * mlp_ratio)
        # MLP 模块
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # 应用权重初始化
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        权重初始化函数
        
        Args:
            m: 网络模块
        """
        # 如果是线性层
        if isinstance(m, nn.Linear):
            # 使用截断正态分布初始化权重
            trunc_normal_(m.weight, std=.02)
            # 如果有偏置项，初始化为 0
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        # 如果是 LayerNorm 层
        elif isinstance(m, nn.LayerNorm):
            # 偏置项初始化为 0
            nn.init.constant_(m.bias, 0)
            # 权重初始化为 1.0
            nn.init.constant_(m.weight, 1.0)
        # 如果是卷积层
        elif isinstance(m, nn.Conv2d):
            # 计算 fan_out（输出连接数）
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            # 使用正态分布初始化权重
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            # 如果有偏置项，初始化为 0
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        """
        前向传播函数
        
        Args:
            x: 输入特征张量 [B, N, C]
            H: 特征图高度
            W: 特征图宽度
        
        Returns:
            输出特征张量
        """
        # 注意力分支：残差连接 + DropPath + 注意力 + LayerNorm
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        # MLP 分支：残差连接 + DropPath + MLP + LayerNorm
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed(nn.Module):
    """
    重叠补丁嵌入模块，将图像转换为补丁嵌入
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_channels=3, embed_dim=768):
        """
        初始化重叠补丁嵌入模块
        
        Args:
            img_size (int): 输入图像尺寸
            patch_size (int): 补丁尺寸
            stride (int): 卷积步长
            in_channels (int): 输入通道数
            embed_dim (int): 嵌入维度
        """
        super().__init__()
        # 将图像尺寸和补丁尺寸转换为二元组
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        # 保存参数
        self.img_size = img_size
        self.patch_size = patch_size
        # 计算补丁网格的高度和宽度
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        # 计算补丁总数
        self.num_patches = self.H * self.W
        # 投影卷积层，将图像补丁映射到嵌入空间
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        # LayerNorm 层
        self.norm = nn.LayerNorm(embed_dim)

        # 应用权重初始化
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        权重初始化函数
        
        Args:
            m: 网络模块
        """
        # 如果是线性层
        if isinstance(m, nn.Linear):
            # 使用截断正态分布初始化权重
            trunc_normal_(m.weight, std=.02)
            # 如果有偏置项，初始化为 0
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        # 如果是 LayerNorm 层
        elif isinstance(m, nn.LayerNorm):
            # 偏置项初始化为 0
            nn.init.constant_(m.bias, 0)
            # 权重初始化为 1.0
            nn.init.constant_(m.weight, 1.0)
        # 如果是卷积层
        elif isinstance(m, nn.Conv2d):
            # 计算 fan_out（输出连接数）
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            # 使用正态分布初始化权重
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            # 如果有偏置项，初始化为 0
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        """
        前向传播函数
        
        Args:
            x: 输入图像张量 [B, C, H, W]
        
        Returns:
            tuple: (补丁嵌入张量 [B, N, C], 特征图高度, 特征图宽度)
        """
        # 应用投影卷积
        x = self.proj(x)
        # 获取特征图的高度和宽度
        _, _, H, W = x.shape
        # 将特征图展平并转置为 [B, N, C] 格式
        x = x.flatten(2).transpose(1, 2)
        # 应用 LayerNorm
        x = self.norm(x)

        return x, H, W


class PyramidVisionTransformerImpr(nn.Module):
    """
    改进的金字塔视觉 Transformer（PVT v2）
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        """
        初始化 PVT v2 模型
        
        Args:
            img_size (int): 输入图像尺寸
            patch_size (int): 补丁尺寸
            in_channels (int): 输入通道数
            num_classes (int): 分类类别数
            embed_dims (list): 各阶段的嵌入维度
            num_heads (list): 各阶段的注意力头数
            mlp_ratios (list): 各阶段的 MLP 比例
            qkv_bias (bool): 是否在 QKV 线性层中使用偏置
            qk_scale (float): QK 缩放因子
            drop_rate (float): Dropout 概率
            attn_drop_rate (float): 注意力 Dropout 概率
            drop_path_rate (float): DropPath 概率
            norm_layer: 归一化层
            depths (list): 各阶段的 Transformer 块数量
            sr_ratios (list): 各阶段的空间缩减比例
        """
        super().__init__()
        # 保存分类类别数和深度
        self.num_classes = num_classes
        self.depths = depths

        # 补丁嵌入层
        # 第一阶段：7x7 卷积，步长 4
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_channels=in_channels,
                                              embed_dim=embed_dims[0])
        # 第二阶段：3x3 卷积，步长 2
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_channels=embed_dims[0],
                                              embed_dim=embed_dims[1])
        # 第三阶段：3x3 卷积，步长 2
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_channels=embed_dims[1],
                                              embed_dim=embed_dims[2])
        # 第四阶段：3x3 卷积，步长 2
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_channels=embed_dims[2],
                                              embed_dim=embed_dims[3])

        # Transformer 编码器
        # 生成随机深度衰减规则
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        # 第一阶段的 Transformer 块
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        # 第一阶段的 LayerNorm
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        # 第二阶段的 Transformer 块
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        # 第二阶段的 LayerNorm
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        # 第三阶段的 Transformer 块
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        # 第三阶段的 LayerNorm
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        # 第四阶段的 Transformer 块
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        # 第四阶段的 LayerNorm
        self.norm4 = norm_layer(embed_dims[3])

        # 分类头（已注释，用于特征提取）
        # self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        # 应用权重初始化
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        权重初始化函数
        
        Args:
            m: 网络模块
        """
        # 如果是线性层
        if isinstance(m, nn.Linear):
            # 使用截断正态分布初始化权重
            trunc_normal_(m.weight, std=.02)
            # 如果有偏置项，初始化为 0
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        # 如果是 LayerNorm 层
        elif isinstance(m, nn.LayerNorm):
            # 偏置项初始化为 0
            nn.init.constant_(m.bias, 0)
            # 权重初始化为 1.0
            nn.init.constant_(m.weight, 1.0)
        # 如果是卷积层
        elif isinstance(m, nn.Conv2d):
            # 计算 fan_out（输出连接数）
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            # 使用正态分布初始化权重
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            # 如果有偏置项，初始化为 0
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self, pretrained=None):
        """
        初始化预训练权重（已注释）
        
        Args:
            pretrained: 预训练权重路径
        """
        if isinstance(pretrained, str):
            logger = 1
            #load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

    def reset_drop_path(self, drop_path_rate):
        """
        重置 DropPath 概率
        
        Args:
            drop_path_rate (float): 新的 DropPath 概率
        """
        # 生成新的随机深度衰减规则
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        # 更新第一阶段的 DropPath 概率
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        # 更新第二阶段的 DropPath 概率
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        # 更新第三阶段的 DropPath 概率
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        # 更新第四阶段的 DropPath 概率
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def freeze_patch_emb(self):
        """
        冻结第一阶段的补丁嵌入层
        """
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        """
        返回不需要权重衰减的参数名称
        """
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}

    def get_classifier(self):
        """
        获取分类器
        """
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        """
        重置分类器
        
        Args:
            num_classes (int): 新的分类类别数
            global_pool (str): 全局池化方式
        """
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        """
        特征提取前向传播
        
        Args:
            x: 输入图像张量 [B, C, H, W]
        
        Returns:
            list: 各阶段的特征图列表
        """
        # 获取批次大小
        B = x.shape[0]
        # 存储各阶段输出
        outs = []

        # 第一阶段
        # 补丁嵌入
        x, H, W = self.patch_embed1(x)
        # 通过 Transformer 块
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        # 应用 LayerNorm
        x = self.norm1(x)
        # 重塑为图像格式并添加到输出列表
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # 第二阶段
        # 补丁嵌入
        x, H, W = self.patch_embed2(x)
        # 通过 Transformer 块
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        # 应用 LayerNorm
        x = self.norm2(x)
        # 重塑为图像格式并添加到输出列表
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # 第三阶段
        # 补丁嵌入
        x, H, W = self.patch_embed3(x)
        # 通过 Transformer 块
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        # 应用 LayerNorm
        x = self.norm3(x)
        # 重塑为图像格式并添加到输出列表
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # 第四阶段
        # 补丁嵌入
        x, H, W = self.patch_embed4(x)
        # 通过 Transformer 块
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        # 应用 LayerNorm
        x = self.norm4(x)
        # 重塑为图像格式并添加到输出列表
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs

        # return x.mean(dim=1)

    def forward(self, x):
        """
        前向传播函数
        
        Args:
            x: 输入图像张量 [B, C, H, W]
        
        Returns:
            各阶段的特征图列表
        """
        # 提取特征
        x = self.forward_features(x)
        # x = self.head(x)

        return x


class DWConv(nn.Module):
    """
    深度可分离卷积模块
    """
    def __init__(self, dim=768):
        """
        初始化深度可分离卷积
        
        Args:
            dim (int): 输入特征维度
        """
        super(DWConv, self).__init__()
        # 深度可分离卷积层：3x3 卷积，groups=dim 实现深度可分离
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        """
        前向传播函数
        
        Args:
            x: 输入特征张量 [B, N, C]
            H: 特征图高度
            W: 特征图宽度
        
        Returns:
            输出特征张量 [B, N, C]
        """
        # 获取输入张量的形状
        B, N, C = x.shape
        # 将输入重塑为图像格式
        x = x.transpose(1, 2).view(B, C, H, W).contiguous()
        # 应用深度可分离卷积
        x = self.dwconv(x)
        # 重塑回序列格式
        x = x.flatten(2).transpose(1, 2)

        return x


def _conv_filter(state_dict, patch_size=16):
    """
    转换补丁嵌入权重，从手动分块 + 线性投影转换为卷积
    
    Args:
        state_dict (dict): 状态字典
        patch_size (int): 补丁尺寸
    
    Returns:
        dict: 转换后的状态字典
    """
    out_dict = {}
    for k, v in state_dict.items():
        # 如果是补丁嵌入投影权重
        if 'patch_embed.proj.weight' in k:
            # 重塑为卷积权重格式
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict


class pvt_v2_b0(PyramidVisionTransformerImpr):
    """
    PVT v2 B0 模型（最小版本）
    """
    def __init__(self, **kwargs):
        """
        初始化 PVT v2 B0 模型
        """
        super(pvt_v2_b0, self).__init__(
            patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class pvt_v2_b1(PyramidVisionTransformerImpr):
    """
    PVT v2 B1 模型（小版本）
    """
    def __init__(self, **kwargs):
        """
        初始化 PVT v2 B1 模型
        """
        super(pvt_v2_b1, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class pvt_v2_b2(PyramidVisionTransformerImpr):
    """
    PVT v2 B2 模型（中等版本）
    """
    def __init__(self, in_channels=3, **kwargs):
        """
        初始化 PVT v2 B2 模型
        
        Args:
            in_channels (int): 输入通道数
        """
        super(pvt_v2_b2, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, in_channels=in_channels)


class pvt_v2_b3(PyramidVisionTransformerImpr):
    """
    PVT v2 B3 模型（大版本）
    """
    def __init__(self, **kwargs):
        """
        初始化 PVT v2 B3 模型
        """
        super(pvt_v2_b3, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class pvt_v2_b4(PyramidVisionTransformerImpr):
    """
    PVT v2 B4 模型（超大版本）
    """
    def __init__(self, **kwargs):
        """
        初始化 PVT v2 B4 模型
        """
        super(pvt_v2_b4, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class pvt_v2_b5(PyramidVisionTransformerImpr):
    """
    PVT v2 B5 模型（最大版本）
    """
    def __init__(self, **kwargs):
        """
        初始化 PVT v2 B5 模型
        """
        super(pvt_v2_b5, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)
