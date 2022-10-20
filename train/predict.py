import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model.createmodel import resnest


def main(myfile):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    root = "/home/junnkidesu/dataset/posiontest"
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    supported = [".jpg", ".JPG", ".png", ".PNG"]
    val_images_path = []  # 存储验证集的所有图片路径
    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # net1 create model
    net1 = resnest(device,numclass=5).to(device)
    # load model weights
    # net1_weight_path = "./posion.pth"
    # net1.load_state_dict(torch.load(net1_weight_path, map_location=device))

    # net2 create model
    net2 = resnest(device,numclass=5).to(device)
    # load model weights
    # net2_weight_path = "./posion.pth"
    # net2.load_state_dict(torch.load(net2_weight_path, map_location=device))

    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]

        for img_path in images:
            val_images_path.append(img_path)
    for val_path in val_images_path:
        img_path1=val_path
        assert os.path.exists(img_path1), "file: '{}' dose not exist.".format(img_path1)
        img = Image.open(img_path1).convert('RGB')
        # plt.imshow(img)
        # [N, C, H, W]
        img = data_transform(img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        net1.eval()
        net2.eval()
        with torch.no_grad():
            # predict class
            output1 = torch.squeeze(net1(img.to(device))).cpu()
            output2 = torch.squeeze(net2(img.to(device))).cpu()
            output_com = output1 + output2
            predict = torch.softmax(output_com, dim=0)
            predict_cla = torch.argmax(predict).numpy()

        print(val_path)
        print_res = "predict class: {}\ttrue class: {}\tprob: {:.3}".format(class_indict[str(predict_cla)],val_path.split('/')[5],
                                                     predict[predict_cla].numpy())
        # plt.title(print_res)
        p = int(class_indict[str(predict_cla)])
        t = int(val_path.split('/')[5])
        ju = 0
        if p==t:
            ju = 1
        else:
            ju = 0
        print(print_res +"\t"+str(ju))
        res = val_path+"\t"+print_res+"\t"+str(ju)+"\n"
        myfile.write(res)
        # plt.show()


if __name__ == '__main__':
    myfile = open(
        '/home/junnkidesu/Downloads/test1.csv',
        'w')
    main(myfile)
