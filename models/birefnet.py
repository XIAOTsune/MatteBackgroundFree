# 导入 PyTorch 核心库
import torch
# 导入 PyTorch 神经网络模块
import torch.nn as nn
# 导入 PyTorch 函数式接口
import torch.nn.functional as F
# 导入 einops 库用于张量重排操作
from einops import rearrange
# 导入 kornia 库中的拉普拉斯滤波器
from kornia.filters import laplacian
# 导入 HuggingFace Hub 的 PyTorch 模型混合类
from huggingface_hub import PyTorchModelHubMixin

# 从本地模块导入配置类
from config import Config
# 从数据集模块导入分类标签
from dataset import class_labels_TR_sorted
# 从骨干网络构建模块导入构建函数
from models.backbones.build_backbone import build_backbone
# 从解码器块模块导入基础解码块和残差块
from models.modules.decoder_blocks import BasicDecBlk, ResBlk
# 从侧向块模块导入基础侧向块
from models.modules.lateral_blocks import BasicLatBlk
# 从 ASPP 模块导入空洞空间金字塔池化
from models.modules.aspp import ASPP, ASPPDeformable
# 从精细化模块导入精细化器
from models.refinement.refiner import Refiner, RefinerPVTInChannels4, RefUNet
# 从干层模块导入干层
from models.refinement.stem_layer import StemLayer


# 定义图像转补丁函数，将图像分割成多个补丁
def image2patches(image, grid_h=2, grid_w=2, patch_ref=None, transformation='b c (hg h) (wg w) -> (b hg wg) c h w'):
    # 如果提供了参考补丁，根据参考补丁计算网格大小
    if patch_ref is not None:
        grid_h, grid_w = image.shape[-2] // patch_ref.shape[-2], image.shape[-1] // patch_ref.shape[-1]
    # 使用 einops 重排张量，将图像分割成补丁
    patches = rearrange(image, transformation, hg=grid_h, wg=grid_w)
    # 返回补丁
    return patches

# 定义补丁转图像函数，将多个补丁合并成图像
def patches2image(patches, grid_h=2, grid_w=2, patch_ref=None, transformation='(b hg wg) c h w -> b c (hg h) (wg w)'):
    # 如果提供了参考补丁，根据参考补丁计算网格大小
    if patch_ref is not None:
        grid_h, grid_w = patch_ref.shape[-2] // patches[0].shape[-2], patch_ref.shape[-1] // patches[0].shape[-1]
    # 使用 einops 重排张量，将补丁合并成图像
    image = rearrange(patches, transformation, hg=grid_h, wg=grid_w)
    # 返回图像
    return image

# 定义 BiRefNet 类，继承自 nn.Module 和 PyTorchModelHubMixin
class BiRefNet(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="birefnet",
    repo_url="https://github.com/ZhengPeng7/BiRefNet",
    tags=['Image Segmentation', 'Background Removal', 'Mask Generation', 'Dichotomous Image Segmentation', 'Camouflaged Object Detection', 'Salient Object Detection']
):
    # 初始化函数
    def __init__(self, bb_pretrained=True):
        # 调用父类初始化函数
        super(BiRefNet, self).__init__()
        # 创建配置实例
        self.config = Config()
        # 设置训练轮次
        self.epoch = 1
        # 构建骨干网络
        self.bb = build_backbone(self.config.bb, pretrained=bb_pretrained)

        # 获取侧向通道配置
        channels = self.config.lateral_channels_in_collection

        # 如果启用辅助分类
        if self.config.auxiliary_classification:
            # 创建自适应平均池化层
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            # 创建分类头
            self.cls_head = nn.Sequential(
                nn.Linear(channels[0], len(class_labels_TR_sorted))
            )

        # 如果启用压缩块
        if self.config.squeeze_block:
            # 创建压缩模块
            self.squeeze_module = nn.Sequential(*[
                eval(self.config.squeeze_block.split('_x')[0])(channels[0]+sum(self.config.cxt), channels[0])
                for _ in range(eval(self.config.squeeze_block.split('_x')[1]))
            ])

        # 创建解码器
        self.decoder = Decoder(channels)

        # 如果启用结束器
        if self.config.ender:
            # 创建解码结束模块
            self.dec_end = nn.Sequential(
                nn.Conv2d(1, 16, 3, 1, 1),
                nn.Conv2d(16, 1, 3, 1, 1),
                nn.ReLU(inplace=True),
            )

        # refine patch-level segmentation
        # 精细化补丁级分割
        if self.config.refine:
            # 如果精细化方式为自身
            if self.config.refine == 'itself':
                # 创建干层
                self.stem_layer = StemLayer(in_channels=3+1, inter_channels=48, out_channels=3, norm_layer='BN' if self.config.batch_size > 1 else 'LN')
            else:
                # 创建精细化器
                self.refiner = eval('{}({})'.format(self.config.refine, 'in_channels=3+1'))

        # 如果冻结骨干网络
        if self.config.freeze_bb:
            # Freeze the backbone...
            # 冻结骨干网络参数
            print(self.named_parameters())
            # 遍历所有参数
            for key, value in self.named_parameters():
                # 如果参数属于骨干网络且不属于精细化器
                if 'bb.' in key and 'refiner.' not in key:
                    # 设置参数不需要梯度
                    value.requires_grad = False

    # 定义前向编码函数
    def forward_enc(self, x):
        # 如果骨干网络是 VGG 或 ResNet
        if self.config.bb in ['vgg16', 'vgg16bn', 'resnet50']:
            # 逐层前向传播
            x1 = self.bb.conv1(x); x2 = self.bb.conv2(x1); x3 = self.bb.conv3(x2); x4 = self.bb.conv4(x3)
        else:
            # 直接通过骨干网络获取多尺度特征
            x1, x2, x3, x4 = self.bb(x)
            # 如果多尺度输入方式为拼接
            if self.config.mul_scl_ipt == 'cat':
                # 获取输入尺寸
                B, C, H, W = x.shape
                # 对缩小的输入进行前向传播
                x1_, x2_, x3_, x4_ = self.bb(F.interpolate(x, size=(H//2, W//2), mode='bilinear', align_corners=True))
                # 将原始特征和缩小特征拼接
                x1 = torch.cat([x1, F.interpolate(x1_, size=x1.shape[2:], mode='bilinear', align_corners=True)], dim=1)
                x2 = torch.cat([x2, F.interpolate(x2_, size=x2.shape[2:], mode='bilinear', align_corners=True)], dim=1)
                x3 = torch.cat([x3, F.interpolate(x3_, size=x3.shape[2:], mode='bilinear', align_corners=True)], dim=1)
                x4 = torch.cat([x4, F.interpolate(x4_, size=x4.shape[2:], mode='bilinear', align_corners=True)], dim=1)
            # 如果多尺度输入方式为相加
            elif self.config.mul_scl_ipt == 'add':
                # 获取输入尺寸
                B, C, H, W = x.shape
                # 对缩小的输入进行前向传播
                x1_, x2_, x3_, x4_ = self.bb(F.interpolate(x, size=(H//2, W//2), mode='bilinear', align_corners=True))
                # 将原始特征和缩小特征相加
                x1 = x1 + F.interpolate(x1_, size=x1.shape[2:], mode='bilinear', align_corners=True)
                x2 = x2 + F.interpolate(x2_, size=x2.shape[2:], mode='bilinear', align_corners=True)
                x3 = x3 + F.interpolate(x3_, size=x3.shape[2:], mode='bilinear', align_corners=True)
                x4 = x4 + F.interpolate(x4_, size=x4.shape[2:], mode='bilinear', align_corners=True)
        # 如果在训练模式且启用辅助分类，计算分类预测
        class_preds = self.cls_head(self.avgpool(x4).view(x4.shape[0], -1)) if self.training and self.config.auxiliary_classification else None
        # 如果启用上下文特征
        if self.config.cxt:
            # 将多尺度特征拼接到最深层特征
            x4 = torch.cat(
                (
                    *[
                        F.interpolate(x1, size=x4.shape[2:], mode='bilinear', align_corners=True),
                        F.interpolate(x2, size=x4.shape[2:], mode='bilinear', align_corners=True),
                        F.interpolate(x3, size=x4.shape[2:], mode='bilinear', align_corners=True),
                    ][-len(self.config.cxt):],
                    x4
                ),
                dim=1
            )
        # 返回多尺度特征和分类预测
        return (x1, x2, x3, x4), class_preds

    # 定义原始前向传播函数
    def forward_ori(self, x):
        ########## Encoder ##########
        # 编码器前向传播
        (x1, x2, x3, x4), class_preds = self.forward_enc(x)
        # 如果启用压缩块
        if self.config.squeeze_block:
            # 对最深层特征进行压缩
            x4 = self.squeeze_module(x4)
        ########## Decoder ##########
        # 解码器前向传播
        features = [x, x1, x2, x3, x4]
        # 如果在训练模式且启用输出参考
        if self.training and self.config.out_ref:
            # 添加拉普拉斯边缘特征
            features.append(laplacian(torch.mean(x, dim=1).unsqueeze(1), kernel_size=5))
        # 通过解码器获取多尺度预测
        scaled_preds = self.decoder(features)
        # 返回多尺度预测和分类预测
        return scaled_preds, class_preds

    # 定义前向传播函数
    def forward(self, x):
        # 调用原始前向传播函数
        scaled_preds, class_preds = self.forward_ori(x)
        # 将分类预测包装成列表
        class_preds_lst = [class_preds]
        # 如果在训练模式，返回预测和分类结果；否则只返回预测结果
        return [scaled_preds, class_preds_lst] if self.training else scaled_preds


# 定义解码器类
class Decoder(nn.Module):
    # 初始化函数
    def __init__(self, channels):
        # 调用父类初始化函数
        super(Decoder, self).__init__()
        # 创建配置实例
        self.config = Config()
        # 动态获取解码器块类
        DecoderBlock = eval(self.config.dec_blk)
        # 动态获取侧向块类
        LateralBlock = eval(self.config.lat_blk)

        # 如果启用解码器输入
        if self.config.dec_ipt:
            # 设置分割标志
            self.split = self.config.dec_ipt_split
            # 设置解码器输入通道数
            N_dec_ipt = 64
            # 设置解码器块类
            DBlock = SimpleConvs
            # 设置中间通道数
            ic = 64
            # 设置输入通道选项
            ipt_cha_opt = 1
            # 创建各层输入块
            self.ipt_blk5 = DBlock(2**10*3 if self.split else 3, [N_dec_ipt, channels[0]//8][ipt_cha_opt], inter_channels=ic)
            self.ipt_blk4 = DBlock(2**8*3 if self.split else 3, [N_dec_ipt, channels[0]//8][ipt_cha_opt], inter_channels=ic)
            self.ipt_blk3 = DBlock(2**6*3 if self.split else 3, [N_dec_ipt, channels[1]//8][ipt_cha_opt], inter_channels=ic)
            self.ipt_blk2 = DBlock(2**4*3 if self.split else 3, [N_dec_ipt, channels[2]//8][ipt_cha_opt], inter_channels=ic)
            self.ipt_blk1 = DBlock(2**0*3 if self.split else 3, [N_dec_ipt, channels[3]//8][ipt_cha_opt], inter_channels=ic)
        else:
            # 不启用分割
            self.split = None

        # 创建解码器块
        self.decoder_block4 = DecoderBlock(channels[0]+([N_dec_ipt, channels[0]//8][ipt_cha_opt] if self.config.dec_ipt else 0), channels[1])
        self.decoder_block3 = DecoderBlock(channels[1]+([N_dec_ipt, channels[0]//8][ipt_cha_opt] if self.config.dec_ipt else 0), channels[2])
        self.decoder_block2 = DecoderBlock(channels[2]+([N_dec_ipt, channels[1]//8][ipt_cha_opt] if self.config.dec_ipt else 0), channels[3])
        self.decoder_block1 = DecoderBlock(channels[3]+([N_dec_ipt, channels[2]//8][ipt_cha_opt] if self.config.dec_ipt else 0), channels[3]//2)
        # 创建输出卷积层
        self.conv_out1 = nn.Sequential(nn.Conv2d(channels[3]//2+([N_dec_ipt, channels[3]//8][ipt_cha_opt] if self.config.dec_ipt else 0), 1, 1, 1, 0))

        # 创建侧向块
        self.lateral_block4 = LateralBlock(channels[1], channels[1])
        self.lateral_block3 = LateralBlock(channels[2], channels[2])
        self.lateral_block2 = LateralBlock(channels[3], channels[3])

        # 如果启用多尺度监督
        if self.config.ms_supervision:
            # 创建多尺度监督卷积层
            self.conv_ms_spvn_4 = nn.Conv2d(channels[1], 1, 1, 1, 0)
            self.conv_ms_spvn_3 = nn.Conv2d(channels[2], 1, 1, 1, 0)
            self.conv_ms_spvn_2 = nn.Conv2d(channels[3], 1, 1, 1, 0)

            # 如果启用输出参考
            if self.config.out_ref:
                # 设置梯度特征通道数
                _N = 16
                # 创建梯度卷积层
                self.gdt_convs_4 = nn.Sequential(nn.Conv2d(channels[1], _N, 3, 1, 1), nn.BatchNorm2d(_N) if self.config.batch_size > 1 else nn.Identity(), nn.ReLU(inplace=True))
                self.gdt_convs_3 = nn.Sequential(nn.Conv2d(channels[2], _N, 3, 1, 1), nn.BatchNorm2d(_N) if self.config.batch_size > 1 else nn.Identity(), nn.ReLU(inplace=True))
                self.gdt_convs_2 = nn.Sequential(nn.Conv2d(channels[3], _N, 3, 1, 1), nn.BatchNorm2d(_N) if self.config.batch_size > 1 else nn.Identity(), nn.ReLU(inplace=True))

                # 创建梯度预测卷积层
                self.gdt_convs_pred_4 = nn.Sequential(nn.Conv2d(_N, 1, 1, 1, 0))
                self.gdt_convs_pred_3 = nn.Sequential(nn.Conv2d(_N, 1, 1, 1, 0))
                self.gdt_convs_pred_2 = nn.Sequential(nn.Conv2d(_N, 1, 1, 1, 0))
                
                # 创建梯度注意力卷积层
                self.gdt_convs_attn_4 = nn.Sequential(nn.Conv2d(_N, 1, 1, 1, 0))
                self.gdt_convs_attn_3 = nn.Sequential(nn.Conv2d(_N, 1, 1, 1, 0))
                self.gdt_convs_attn_2 = nn.Sequential(nn.Conv2d(_N, 1, 1, 1, 0))

    # 定义前向传播函数
    def forward(self, features):
        # 如果在训练模式且启用输出参考
        if self.training and self.config.out_ref:
            # 初始化梯度预测和标签输出列表
            outs_gdt_pred = []
            outs_gdt_label = []
            # 解包特征，包含梯度真值
            x, x1, x2, x3, x4, gdt_gt = features
        else:
            # 解包特征，不包含梯度真值
            x, x1, x2, x3, x4 = features
        # 初始化输出列表
        outs = []

        # 如果启用解码器输入
        if self.config.dec_ipt:
            # 根据分割标志处理输入补丁
            patches_batch = image2patches(x, patch_ref=x4, transformation='b c (hg h) (wg w) -> b (c hg wg) h w') if self.split else x
            # 将输入特征与处理后的补丁拼接
            x4 = torch.cat((x4, self.ipt_blk5(F.interpolate(patches_batch, size=x4.shape[2:], mode='bilinear', align_corners=True))), 1)
        # 通过第4层解码器块
        p4 = self.decoder_block4(x4)
        # 如果启用多尺度监督且在训练模式，计算第4层监督输出
        m4 = self.conv_ms_spvn_4(p4) if self.config.ms_supervision and self.training else None
        # 如果启用输出参考
        if self.config.out_ref:
            # 计算第4层梯度特征
            p4_gdt = self.gdt_convs_4(p4)
            # 如果在训练模式
            if self.training:
                # >> GT:
                # 计算梯度标签
                m4_dia = m4
                gdt_label_main_4 = gdt_gt * F.interpolate(m4_dia, size=gdt_gt.shape[2:], mode='bilinear', align_corners=True)
                outs_gdt_label.append(gdt_label_main_4)
                # >> Pred:
                # 计算梯度预测
                gdt_pred_4 = self.gdt_convs_pred_4(p4_gdt)
                outs_gdt_pred.append(gdt_pred_4)
            # 计算梯度注意力
            gdt_attn_4 = self.gdt_convs_attn_4(p4_gdt).sigmoid()
            # >> Finally:
            # 应用梯度注意力
            p4 = p4 * gdt_attn_4
        # 上采样第4层特征
        _p4 = F.interpolate(p4, size=x3.shape[2:], mode='bilinear', align_corners=True)
        # 与第3层侧向特征相加
        _p3 = _p4 + self.lateral_block4(x3)

        # 如果启用解码器输入
        if self.config.dec_ipt:
            # 根据分割标志处理输入补丁
            patches_batch = image2patches(x, patch_ref=_p3, transformation='b c (hg h) (wg w) -> b (c hg wg) h w') if self.split else x
            # 将特征与处理后的补丁拼接
            _p3 = torch.cat((_p3, self.ipt_blk4(F.interpolate(patches_batch, size=x3.shape[2:], mode='bilinear', align_corners=True))), 1)
        # 通过第3层解码器块
        p3 = self.decoder_block3(_p3)
        # 如果启用多尺度监督且在训练模式，计算第3层监督输出
        m3 = self.conv_ms_spvn_3(p3) if self.config.ms_supervision and self.training else None
        # 如果启用输出参考
        if self.config.out_ref:
            # 计算第3层梯度特征
            p3_gdt = self.gdt_convs_3(p3)
            # 如果在训练模式
            if self.training:
                # >> GT:
                # m3 --dilation--> m3_dia
                # G_3^gt * m3_dia --> G_3^m, which is the label of gradient
                # 计算梯度标签
                m3_dia = m3
                gdt_label_main_3 = gdt_gt * F.interpolate(m3_dia, size=gdt_gt.shape[2:], mode='bilinear', align_corners=True)
                outs_gdt_label.append(gdt_label_main_3)
                # >> Pred:
                # p3 --conv--BN--> F_3^G, where F_3^G predicts the \hat{G_3} with xx
                # F_3^G --sigmoid--> A_3^G
                # 计算梯度预测
                gdt_pred_3 = self.gdt_convs_pred_3(p3_gdt)
                outs_gdt_pred.append(gdt_pred_3)
            # 计算梯度注意力
            gdt_attn_3 = self.gdt_convs_attn_3(p3_gdt).sigmoid()
            # >> Finally:
            # p3 = p3 * A_3^G
            # 应用梯度注意力
            p3 = p3 * gdt_attn_3
        # 上采样第3层特征
        _p3 = F.interpolate(p3, size=x2.shape[2:], mode='bilinear', align_corners=True)
        # 与第2层侧向特征相加
        _p2 = _p3 + self.lateral_block3(x2)

        # 如果启用解码器输入
        if self.config.dec_ipt:
            # 根据分割标志处理输入补丁
            patches_batch = image2patches(x, patch_ref=_p2, transformation='b c (hg h) (wg w) -> b (c hg wg) h w') if self.split else x
            # 将特征与处理后的补丁拼接
            _p2 = torch.cat((_p2, self.ipt_blk3(F.interpolate(patches_batch, size=x2.shape[2:], mode='bilinear', align_corners=True))), 1)
        # 通过第2层解码器块
        p2 = self.decoder_block2(_p2)
        # 如果启用多尺度监督且在训练模式，计算第2层监督输出
        m2 = self.conv_ms_spvn_2(p2) if self.config.ms_supervision and self.training else None
        # 如果启用输出参考
        if self.config.out_ref:
            # 计算第2层梯度特征
            p2_gdt = self.gdt_convs_2(p2)
            # 如果在训练模式
            if self.training:
                # >> GT:
                # 计算梯度标签
                m2_dia = m2
                gdt_label_main_2 = gdt_gt * F.interpolate(m2_dia, size=gdt_gt.shape[2:], mode='bilinear', align_corners=True)
                outs_gdt_label.append(gdt_label_main_2)
                # >> Pred:
                # 计算梯度预测
                gdt_pred_2 = self.gdt_convs_pred_2(p2_gdt)
                outs_gdt_pred.append(gdt_pred_2)
            # 计算梯度注意力
            gdt_attn_2 = self.gdt_convs_attn_2(p2_gdt).sigmoid()
            # >> Finally:
            # 应用梯度注意力
            p2 = p2 * gdt_attn_2
        # 上采样第2层特征
        _p2 = F.interpolate(p2, size=x1.shape[2:], mode='bilinear', align_corners=True)
        # 与第1层侧向特征相加
        _p1 = _p2 + self.lateral_block2(x1)

        # 如果启用解码器输入
        if self.config.dec_ipt:
            # 根据分割标志处理输入补丁
            patches_batch = image2patches(x, patch_ref=_p1, transformation='b c (hg h) (wg w) -> b (c hg wg) h w') if self.split else x
            # 将特征与处理后的补丁拼接
            _p1 = torch.cat((_p1, self.ipt_blk2(F.interpolate(patches_batch, size=x1.shape[2:], mode='bilinear', align_corners=True))), 1)
        # 通过第1层解码器块
        _p1 = self.decoder_block1(_p1)
        # 上采样到原始输入尺寸
        _p1 = F.interpolate(_p1, size=x.shape[2:], mode='bilinear', align_corners=True)

        # 如果启用解码器输入
        if self.config.dec_ipt:
            # 根据分割标志处理输入补丁
            patches_batch = image2patches(x, patch_ref=_p1, transformation='b c (hg h) (wg w) -> b (c hg wg) h w') if self.split else x
            # 将特征与处理后的补丁拼接
            _p1 = torch.cat((_p1, self.ipt_blk1(F.interpolate(patches_batch, size=x.shape[2:], mode='bilinear', align_corners=True))), 1)
        # 通过输出卷积层得到最终预测
        p1_out = self.conv_out1(_p1)

        # 如果启用多尺度监督且在训练模式
        if self.config.ms_supervision and self.training:
            # 添加多尺度监督输出
            outs.append(m4)
            outs.append(m3)
            outs.append(m2)
        # 添加最终输出
        outs.append(p1_out)
        # 根据是否启用输出参考返回不同格式的结果
        return outs if not (self.config.out_ref and self.training) else ([outs_gdt_pred, outs_gdt_label], outs)


# 定义简单卷积类
class SimpleConvs(nn.Module):
    # 初始化函数
    def __init__(
        self, in_channels: int, out_channels: int, inter_channels=64
    ) -> None:
        # 调用父类初始化函数
        super().__init__()
        # 创建第一个卷积层
        self.conv1 = nn.Conv2d(in_channels, inter_channels, 3, 1, 1)
        # 创建输出卷积层
        self.conv_out = nn.Conv2d(inter_channels, out_channels, 3, 1, 1)

    # 定义前向传播函数
    def forward(self, x):
        # 通过两个卷积层
        return self.conv_out(self.conv1(x))


###########


# 定义 BiRefNet 粗到细版本类
class BiRefNetC2F(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="birefnet_c2f",
    repo_url="https://github.com/ZhengPeng7/BiRefNet_C2F",
    tags=['Image Segmentation', 'Background Removal', 'Mask Generation', 'Dichotomous Image Segmentation', 'Camouflaged Object Detection', 'Salient Object Detection']
):
    # 初始化函数
    def __init__(self, bb_pretrained=True):
        # 调用父类初始化函数
        super(BiRefNetC2F, self).__init__()
        # 创建配置实例
        self.config = Config()
        # 设置训练轮次
        self.epoch = 1
        # 设置网格大小
        self.grid = 4
        # 创建粗糙模型
        self.model_coarse = BiRefNet(bb_pretrained=True)
        # 创建精细模型
        self.model_fine = BiRefNet(bb_pretrained=True)
        # 创建输入混合器
        self.input_mixer = nn.Conv2d(4, 3, 1, 1, 0)
        # 创建输出混合器
        self.output_mixer_merge_post = nn.Sequential(nn.Conv2d(1, 16, 3, 1, 1), nn.Conv2d(16, 1, 3, 1, 1))

    # 定义前向传播函数
    def forward(self, x):
        # 克隆原始输入
        x_ori = x.clone()
        ########## Coarse ##########
        # 粗糙阶段：缩小输入尺寸
        x = F.interpolate(x, size=[s//self.grid for s in self.config.size[::-1]], mode='bilinear', align_corners=True)

        # 如果在训练模式
        if self.training:
            # 通过粗糙模型获取预测和分类结果
            scaled_preds, class_preds_lst = self.model_coarse(x)
        else:
            # 只获取预测结果
            scaled_preds = self.model_coarse(x)
        ##########  Fine  ##########
        # 精细阶段：将原始输入分割成补丁
        x_HR_patches = image2patches(x_ori, patch_ref=x, transformation='b c (hg h) (wg w) -> (b hg wg) c h w')
        # 获取粗糙预测结果并上采样到原始尺寸
        pred = F.interpolate(scaled_preds[-1] if not (self.config.out_ref and self.training) else scaled_preds[1][-1], size=x_ori.shape[2:], mode='bilinear', align_corners=True)
        # 将预测结果分割成补丁
        pred_patches = image2patches(pred, patch_ref=x, transformation='b c (hg h) (wg w) -> (b hg wg) c h w')
        # 将高分辨率图像补丁和预测补丁拼接
        t = torch.cat([x_HR_patches, pred_patches], dim=1)
        # 通过输入混合器处理
        x_HR = self.input_mixer(t)

        # 将预测结果重新排列为补丁格式
        pred_patches = image2patches(pred, patch_ref=x_HR, transformation='b c (hg h) (wg w) -> b (c hg wg) h w')
        # 如果在训练模式
        if self.training:
            # 通过精细模型获取预测和分类结果
            scaled_preds_HR, class_preds_lst_HR = self.model_fine(x_HR)
        else:
            # 只获取预测结果
            scaled_preds_HR = self.model_fine(x_HR)
        # 如果在训练模式
        if self.training:
            # 如果启用输出参考
            if self.config.out_ref:
                # 解包粗糙预测结果
                [outs_gdt_pred, outs_gdt_label], outs = scaled_preds
                # 解包精细预测结果
                [outs_gdt_pred_HR, outs_gdt_label_HR], outs_HR = scaled_preds_HR
                # 将精细预测补丁合并成完整图像
                for idx_out, out_HR in enumerate(outs_HR):
                    outs_HR[idx_out] = self.output_mixer_merge_post(patches2image(out_HR, grid_h=self.grid, grid_w=self.grid, transformation='(b hg wg) c h w -> b c (hg h) (wg w)'))
                # 返回合并的梯度预测、标签和输出，以及分类预测
                return [([outs_gdt_pred + outs_gdt_pred_HR, outs_gdt_label + outs_gdt_label_HR], outs + outs_HR), class_preds_lst]    # handle gt here
            else:
                # 返回合并的预测结果和分类预测
                return [
                    scaled_preds + [self.output_mixer_merge_post(patches2image(scaled_pred_HR, grid_h=self.grid, grid_w=self.grid, transformation='(b hg wg) c h w -> b c (hg h) (wg w)')) for scaled_pred_HR in scaled_preds_HR],
                    class_preds_lst
                ]
        else:
            # 返回合并的预测结果
            return scaled_preds + [self.output_mixer_merge_post(patches2image(scaled_pred_HR, grid_h=self.grid, grid_w=self.grid, transformation='(b hg wg) c h w -> b c (hg h) (wg w)')) for scaled_pred_HR in scaled_preds_HR]
