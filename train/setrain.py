import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import pretrainedmodels
# from model.re import resnet34, resnet50, resnet50_32x4d
# from se_resnet import se_resnet50
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def setrain():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    # 实例化SummaryWriter对象
    # tb_writer = SummaryWriter(log_dir="runs/se_resnext50")

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    image_path1 = '/home/junnkidesu/dataset'
    assert os.path.exists(image_path1), "{} path does not exist.".format(image_path1)
    train_dataset1 = datasets.ImageFolder(root=os.path.join(image_path1, "lesstrain"),
                                         transform=data_transform["train"])


    batch_size = 64
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader1 = torch.utils.data.DataLoader(train_dataset1,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset1 = datasets.ImageFolder(root=os.path.join(image_path1, "lessval"),
                                            transform=data_transform["val"])
    validate_loader1 = torch.utils.data.DataLoader(validate_dataset1,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    # set train label
    labelroot1 = '/home/junnkidesu/oschangepy/label3.txt'
    train_label1 = []
    with open(labelroot1, "r") as f:
        for i, line in enumerate(f.readlines()):
            line = line.strip('\n')
            name1 = line.split(',')[-1]
            train_label1.append(int(name1))

    net1 = pretrainedmodels.__dict__['se_resnext50_32x4d'](num_classes=1000, pretrained=None)
    model_weight_path1 = "./se_resnext50_32x4d-a260b3a4.pth"
    assert os.path.exists(model_weight_path1), "file {} does not exist.".format(model_weight_path1)
    net1.load_state_dict(torch.load(model_weight_path1, map_location=device))
    model = nn.Linear(2048, 708)

    net1.last_linear = model

    net1.to(device)

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net1.parameters() if p.requires_grad]
    optimizer1 = optim.Adam(params, lr=0.0001)

    net1.train()
    running_loss = 0.0
    train_bar = tqdm(train_loader1)
    for step, data in enumerate(train_bar):
        images, labels = data
        optimizer1.zero_grad()
        logits = net1(images.to(device))
        loss = loss_function(logits, labels.to(device))
        loss.backward()
        optimizer1.step()

        # print statistics
        running_loss += loss.item()


        cout1 = torch.zeros(709)
        allcout1 = np.array(cout1).astype(int)

        # validate
        net1.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader1, colour='green')
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net1(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                preds1 = predict_y.cpu().numpy()
                label1 = val_labels.cpu().numpy()
                total_preds = np.array(preds1).astype(int)
                total_labels = np.array(label1).astype(int)

        for p, k in zip(total_preds, total_labels):
            if (p == k):
                allcout1[p] += 1
        medium_acc = 0
        medium_num = 0
        few_acc = 0
        few_num = 0
        for p, k in zip(train_label1, allcout1):
            if (p < 6):
                few_acc += k
                few_num += p
            else:
                medium_acc += k
                medium_num += p

    return medium_acc/medium_num,few_acc/few_num,acc


