import argparse
import torch
import torch.utils.data as Data
from torch.autograd import Variable
import os
import numpy as np
from model import FaceRecognition
from dataset import get_dataset

parser = argparse.ArgumentParser(description='face recognition')
parser.add_argument('--batch_size', type=int, default=80, help='training batch size')
parser.add_argument('--num_epochs', type=int, default=20, help='number of training epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--cuda', type=bool, default=True, help='use cuda?')
parser.add_argument('--resume', type=str, default='', help='path to network checkpoint')
parser.add_argument('--start_epoch', type=int, default=0, help='restart epoch number for training')
parser.add_argument('--threads', type=int, default=0, help='number of threads')
parser.add_argument('--pretrained', type=str, default='', help='path to network parameters')
parser.add_argument('--num_channels', type=int, default=3)
parser.add_argument('--train_dir', type=str, default='./data/train', help='LR image path to training data directory')
parser.add_argument('--test_dir', type=str, default='./data/test', help='image path to testing data directory')
opt = parser.parse_args()


# 训练
def train(train_dataloader, test_dataloader, network, optimizer, loss_func):
    print('==> Training...')
    for epoch in range(opt.start_epoch, opt.num_epochs + 1):
        train_process(train_dataloader, network, optimizer, loss_func, epoch, epochs=opt.num_epochs)
        test(test_dataloader, network)
        save_checkpoint(network, epoch)


# 测试
def test(test_dataloader, network):
    print('==> Testing...')
    acc = test_process(test_dataloader, network)
    with open("acc.txt", "a+", encoding="UTF-8") as f:
        f.write("acc: {:.10f}\n".format(acc))


# 每个epoch的训练程序
def train_process(dataloader, network, optimizer, loss_func, epoch=1, epochs=1):
    network.train()
    for iteration, (inputs, labels) in enumerate(dataloader):
        inputs = Variable(inputs)  # 输入数据
        labels = Variable(labels)  # label
        if opt.cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()
        preds = network(inputs)
        loss = loss_func(preds, labels)
        train_acc = calcu_acc(preds, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print('epoch:[{}/{}] batch:[{}/{}] loss:{:.10f} acc:{:.10f}'.format(epoch, epochs, iteration, len(dataloader),
                                                                            loss.data, train_acc))


# 测试程序
def test_process(test_dataloader, network):
    network.eval()
    train_correct = 0
    for idx, (inputs, labels) in enumerate(test_dataloader):
        inputs = Variable(inputs)
        labels = Variable(labels)
        if opt.cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()
        preds = network(inputs)
        train_correct += calcu_acc(preds, labels)
    print(train_correct / len(test_dataloader))
    return train_correct / len(test_dataloader)


def calcu_acc(preds, labels):
    num = 0
    for i in range(len(labels)):
        a = labels[i].cpu().detach().numpy()
        b = np.argmax(preds[i].cpu().detach().numpy())
        if a == b:
            # print(a, b+1)
            num += 1
    return num / len(labels)


# 设计自适应的学习率
def adjust_learning_rate(epoch):
    lr = opt.lr * (0.1 ** (epoch // opt.step))
    return lr


def save_checkpoint(network, epoch):
    model_folder = "model_para/"
    param_path = model_folder + "param_epoch{}.pkl".format(epoch)
    state = {"epoch": epoch, "model": network}
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    torch.save(state, param_path)
    print("Checkpoint saved to {}".format(param_path))


if __name__ == "__main__":
    # 构建网络
    print('==> building network...')
    network = FaceRecognition(label_num=394)

    # loss函数
    loss_func = torch.nn.CrossEntropyLoss()

    # 设置GPU
    if opt.cuda and not torch.cuda.is_available():  # 检查是否有GPU
        raise Exception('No GPU found, please run without --cuda')
    print("==> Setting GPU")
    if opt.cuda:
        print('cuda_mode:', opt.cuda)
        network = network.cuda()
        loss_func = loss_func.cuda()

    # 设置优化器函数
    print("==> Setting Optimizer")
    optimizer = torch.optim.Adam(network.parameters(), lr=opt.lr)

    # 判断网络是否已经训练过或者已经训练完成
    if opt.pretrained:  # 训练完成,进行测试
        # 加载测试数据进行测试
        print('==> loading test data...')
        test_dataset = get_dataset(opt.test_dir)
        test_dataloader = Data.DataLoader(dataset=test_dataset, batch_size=opt.batch_size, shuffle=True,
                                          num_workers=opt.threads)
        if os.path.isfile(opt.pretrained):
            print('==> loading model {}'.format(opt.pretrained))
            weights = torch.load(opt.pretrained)
            network.load_state_dict(weights['model'].state_dict())
            # 进行测试
            test(test_dataloader, network)
        else:
            print('==> no network model found at {}'.format(opt.pretrained))
    else:  # 未训练完成，需要进行训练
        # 加载训练数据
        print('==> loading training data...')
        train_dataset = get_dataset(opt.train_dir)
        test_dataset = get_dataset(opt.test_dir)
        train_dataloader = Data.DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True,
                                           num_workers=opt.threads)
        test_dataloader = Data.DataLoader(dataset=test_dataset, batch_size=opt.batch_size, shuffle=False,
                                          num_workers=opt.threads)
        if opt.resume:  # 部分训练，需要重新开始训练
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint['epoch'] + 1
            print('==> start training at epoch {}'.format(opt.start_epoch))
            network.load_state_dict(checkpoint['model'].state_dict())
            print("==> resume Training...")
        train(train_dataloader, test_dataloader, network, optimizer, loss_func)
