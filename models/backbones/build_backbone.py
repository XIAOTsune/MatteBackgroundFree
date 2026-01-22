# 导入 PyTorch 核心库
import torch
# 导入 PyTorch 神经网络模块
import torch.nn as nn
# 导入有序字典，用于构建有序的网络层
from collections import OrderedDict
# 导入 torchvision 中的预训练模型和权重
from torchvision.models import vgg16, vgg16_bn, VGG16_Weights, VGG16_BN_Weights, resnet50, ResNet50_Weights
# 导入自定义的 PVT v2 系列模型
from models.backbones.pvt_v2 import pvt_v2_b0, pvt_v2_b1, pvt_v2_b2, pvt_v2_b5
# 导入自定义的 Swin Transformer v1 系列模型
from models.backbones.swin_v1 import swin_v1_t, swin_v1_s, swin_v1_b, swin_v1_l
# 导入配置文件
from config import Config

# 创建配置实例
config = Config()

def build_backbone(bb_name, pretrained=True, params_settings=''):
    """
    构建骨干网络的函数
    
    Args:
        bb_name (str): 骨干网络名称
        pretrained (bool): 是否使用预训练权重
        params_settings (str): 参数设置字符串
    
    Returns:
        nn.Module: 构建好的骨干网络
    """
    # 如果是 VGG16 网络
    if bb_name == 'vgg16':
        # 加载 VGG16 模型，根据 pretrained 参数决定是否使用预训练权重
        bb_net = list(vgg16(pretrained=VGG16_Weights.DEFAULT if pretrained else None).children())[0]
        # 将 VGG16 的特征提取层按阶段分组，构建有序字典
        bb = nn.Sequential(OrderedDict({'conv1': bb_net[:4], 'conv2': bb_net[4:9], 'conv3': bb_net[9:16], 'conv4': bb_net[16:23]}))
    # 如果是 VGG16 with BatchNorm 网络
    elif bb_name == 'vgg16bn':
        # 加载 VGG16_BN 模型，根据 pretrained 参数决定是否使用预训练权重        
        bb_net = list(vgg16_bn(pretrained=VGG16_BN_Weights.DEFAULT if pretrained else None).children())[0]
        # 将 VGG16_BN 的特征提取层按阶段分组，构建有序字典
        bb = nn.Sequential(OrderedDict({'conv1': bb_net[:6], 'conv2': bb_net[6:13], 'conv3': bb_net[13:23], 'conv4': bb_net[23:33]}))
    # 如果是 ResNet50 网络
    elif bb_name == 'resnet50':
        # 加载 ResNet50 模型，根据 pretrained 参数决定是否使用预训练权重
        bb_net = list(resnet50(pretrained=ResNet50_Weights.DEFAULT if pretrained else None).children())
        # 将 ResNet50 的层按阶段分组，构建有序字典
        bb = nn.Sequential(OrderedDict({'conv1': nn.Sequential(*bb_net[0:3]), 'conv2': bb_net[4], 'conv3': bb_net[5], 'conv4': bb_net[6]}))
    # 对于其他自定义模型（如 PVT、Swin Transformer 等）
    else:
        # 使用 eval 函数动态创建模型实例，传入参数设置
        bb = eval('{}({})'.format(bb_name, params_settings))
        # 如果需要预训练权重
        if pretrained:
            # 加载预训练权重
            bb = load_weights(bb, bb_name)
    # 返回构建好的骨干网络
    return bb

def load_weights(model, model_name):
    """
    加载预训练权重的函数
    
    Args:
        model (nn.Module): 要加载权重的模型
        model_name (str): 模型名称，用于获取权重文件路径
    
    Returns:
        nn.Module: 加载权重后的模型，如果加载失败返回 None
    """
    # 从配置文件中获取权重文件路径，加载权重文件
    save_model = torch.load(config.weights[model_name], map_location='cpu', weights_only=True)
    # 获取当前模型的状态字典
    model_dict = model.state_dict()
    # 筛选出尺寸匹配的权重，如果尺寸不匹配则保持原有权重
    state_dict = {k: v if v.size() == model_dict[k].size() else model_dict[k] for k, v in save_model.items() if k in model_dict.keys()}
    # 处理权重尺寸不匹配的情况，这在修改骨干网络结构时可能发生
    # 如果没有匹配的权重
    if not state_dict:
        # 获取保存模型的所有键
        save_model_keys = list(save_model.keys())
        # 如果只有一个键，可能权重被包装在子项中
        sub_item = save_model_keys[0] if len(save_model_keys) == 1 else None
        # 尝试从子项中加载权重
        state_dict = {k: v if v.size() == model_dict[k].size() else model_dict[k] for k, v in save_model[sub_item].items() if k in model_dict.keys()}
        # 如果仍然没有匹配的权重或没有子项
        if not state_dict or not sub_item:
            # 打印错误信息
            print('Weights are not successully loaded. Check the state dict of weights file.')
            # 返回 None 表示加载失败
            return None
        else:
            # 打印成功信息，说明在哪个子项中找到了正确的权重
            print('Found correct weights in the "{}" item of loaded state_dict.'.format(sub_item))
    # 更新模型字典
    model_dict.update(state_dict)
    # 加载更新后的状态字典到模型
    model.load_state_dict(model_dict)
    # 返回加载权重后的模型
    return model
