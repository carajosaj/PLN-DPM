from .efficientnet import efficientnet_b0 as create_model
import torch
import os
from .seresnet import se_resnext50_32x4d
import torch.nn as nn
from .resnet import resnet50_32x4d
from .resnest import resnest50
from .resnetbest import resnet50
def effcientb0(device,num_classes):
    model = create_model(num_classes=num_classes).to(device)
    weights_dict = torch.load("", map_location=device)
    load_weights_dict = {k: v for k, v in weights_dict.items()
        if model.state_dict()[k].numel() == v.numel()}
    print(model.load_state_dict(load_weights_dict, strict=False))

    # 是否冻结权重
    # if args.freeze_layers:
    #     for name, para in model.named_parameters():
    #         # 除最后一个卷积层和全连接层外，其他权重全部冻结
    #         if ("features.top" not in name) and ("classifier" not in name):
    #             para.requires_grad_(False)
    #         else:
    #             print("training {}".format(name))
    return model
def resnetbest(device,numclass):
    model = resnet50(num_classes=1000)
    model_weight_path = "/home/junnkidesu/mergecode/pth/resnet50-19c8e357.pth"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    model.load_state_dict(torch.load(model_weight_path))
    in_channel = model.fc.in_features
    model.fc = nn.Linear(in_channel, numclass)  # 因为最后只需要划分出200个类别,所以要把原来的全连接层输出改为200
    model.to(device)
    return model

def seresnet(device,numclass):
    model = se_resnext50_32x4d(num_classes=1000)
    model_weight_path = "/home/junnkidesu/mergecode/pth/se_resnext50_32x4d-a260b3a4.pth"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    fc = nn.Linear(2048, numclass)
    model.last_linear = fc
    return model

def resnet(device,numclass):
    model = resnet50_32x4d()
    model_weight_path = "/home/junnkidesu/mergecode/pth/resnext50_32x4d-7cdf4587.pth"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    in_channel = model.fc.in_features
    model.fc = nn.Linear(in_channel, numclass)
    return model

def resnest(device,numclass):
    model = resnest50(pretrained=False)
    model_weight_path = "/home/junnkidesu/mergecode/pth/resnest50-528c19ca.pth"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    # for name, para in model.named_parameters():
    #     if ("fc." not in name):
    #         para.requires_grad_(False)
    #     else:
    #         print("training {}".format(name))
    in_channel = model.inchannel
    model.fc = nn.Linear(in_channel, numclass)
    return model