
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.tensorboard import SummaryWriter
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision import transforms
from model.createmodel import seresnet,resnet,resnest,resnetbest
from utils.utils import *
from utils.lr_scheduler import *
from utils.my_dataset import MyDataSet

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter()
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")
    batch_size = 32
    epochs = 90
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    rate_schedule = np.ones(epochs) * 0.25
    rate_schedule[:10] = np.linspace(0, 0.25, 10)

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    image_path = '/home/junnkidesu/dataset/ip102_v1.1'
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)


    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])

    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    net1 = resnest(device,102)
    nn.init.kaiming_normal_(net1.fc.weight,mode='fan_in')
    net1.to(device)
    # myfile.write(str(net1))

    net2 = resnest(device,102)
    nn.init.xavier_uniform_(net2.fc.weight)
    net2.to(device)
    # myfile.write(str(net2))
    #------------------------------------------------------------


    params1 = [p for p in net1.parameters() if p.requires_grad]
    optimizer1 = optim.SGD(params1, lr=0.002,weight_decay=0.00005,momentum=0.9)
    # optimizer1 = optim.Adam(params1, lr=0.0001)

    params2 = [p for p in net2.parameters() if p.requires_grad]
    optimizer2 = optim.SGD(params2, lr=0.002,weight_decay=0.00005,momentum=0.9)
    # optimizer2 = optim.Adam(params2, lr=0.0001)

    bestacc = 0.0
    bestacc_com = 0.0
    for epoch in range(epochs):
        net1.train()
        optimizer1 = adjust_learning(optimizer1,epoch,net1)
        net2.train()
        optimizer2 = adjust_learning(optimizer2, epoch,net2)
        train(train_loader, epoch, net1, optimizer1, net2, optimizer2,device,rate_schedule)
        # sum1,allcout1 = fewevaluate(net1,validate_loader,device)
        # sum2,allcout2 = fewevaluate(net2,validate_loader,device)
        sum1,sum2,sum_com,MAE = dualevaluate(net1,net2,validate_loader,device)
        acc1 = sum1/len(validate_dataset)
        acc2 = sum2/len(validate_dataset)
        acc_com = sum_com/len(validate_dataset)
        tags = ["acc1","acc2", "acc_com","MAE"]
        tb_writer.add_scalar(tags[0], acc1, epoch)
        tb_writer.add_scalar(tags[1], acc2, epoch)
        tb_writer.add_scalar(tags[2], acc_com, epoch)
        tb_writer.add_scalar(tags[3], MAE, epoch)
        print("epoch "+str(epoch)+"\t"+"acc1:\t"+str(acc1)+"\t"+"acc2:\t"+str(acc2)+"\tacc_com:\t"+str(acc_com)+"\tMAE:\t"+str(MAE)+"\n")

        if acc1<acc2:
            temp = acc2
        else:
            temp = acc1
        if temp>bestacc:
            bestacc = temp
        if acc_com>bestacc_com:
            bestacc_com = acc_com

    print("The bes acc is " + str(bestacc)+"\t"+"The bes acc_com is "+str(bestacc_com)+"\n")

if __name__ == '__main__':
    main()