# Import the neccessary libraries
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
import copy
import time
from PIL import Image

# https://medium.com/@umerfarooq_26378/model-summary-in-pytorch-b5a1e4b64d25
# tf의 summary 같은 기능을 위해서 추가
from torchsummary import summary

# folder에 있는 image 와 label로 데이테셋 가져 오기
from mk_tensor import make_tensor

# 저장을 위해서 선언
from torch.utils.tensorboard import SummaryWriter
import os


# https://data-panic.tistory.com/21
# 위 사이트 참고해서 custom dataset 만듬
class CustomDataset(Dataset):
    def __init__(self, x, y, x_transform):
        self.x = x
        self.y = y
        self.x_transform = x_transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):

        # https://stackoverflow.com/questions/60138697/typeerror-cannot-handle-this-data-type-1-1-3-f4
        # https://discuss.pytorch.org/t/typeerror-pic-should-be-pil-image-or-ndarray-got-class-numpy-ndarray/20134
        # 위 사이트 참고함
        x = Image.fromarray((self.x[item] * 255).astype(np.uint8))
        x = self.x_transform(x)

        # https://discuss.pytorch.org/t/runtimeerror-expected-object-of-scalar-type-long-but-got-scalar-type-float-when-using-crossentropyloss/30542
        # int가 없다고 뜨는 에러 발생해서 찾아본
        y = self.y[item]

        return x, y


def time_check(name, start, end):
    print("{} time : {:0.3f}".format(name, end - start))

# requirement.txt 쉽게 만들기
# https://itinerant.tistory.com/100

if __name__ == '__main__':
    start_time = time.time()
    learning_rate = 0.001
    Epoch_num = 50
    Batch_size = 1024
    Resolution = 36
    save_path = 'save/resnet_18_36x36'

    # tensorboard 사용법 https://tutorials.pytorch.kr/intermediate/tensorboard_tutorial.html
    writer = SummaryWriter(save_path)
    writer.add_scalar('epoch', Epoch_num)
    writer.add_scalar('batch size', Batch_size)
    writer.add_scalar('learning_rate', learning_rate)
    writer.add_scalar('resolution', Resolution)

    tr_transform = transforms.Compose([transforms.Resize((Resolution, Resolution)),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    te_transform = transforms.Compose([transforms.Resize((Resolution, Resolution)),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    # npy 저장 및 불러오기 https://gldmg.tistory.com/43
    # file 존재 유무 https://wikidocs.net/14304
    train_path = 'data/pre_train'
    train_file = 'data/train.npy'

    test_path = 'data/pre_test'
    test_file = 'data/test.npy'

    if os.path.isfile(train_file):
        load = np.load(train_file, allow_pickle=True)
        tr_x, tr_y = load[0], load[1]
    else:
        tr_x, tr_y = make_tensor('data/pre_train')
        np.save(train_file, np.array([tr_x, tr_y]))

    if os.path.isfile(test_file):
        load = np.load(test_file, allow_pickle=True)
        te_x, te_y = load[0], load[1]
    else:
        te_x, te_y = make_tensor('data/pre_test')
        np.save(test_file, np.array([te_x, te_y]))

    # tr_x, tr_y = make_tensor('data/pre_train')
    # te_x, te_y = make_tensor('data/pre_test')

    processing_time = time.time()
    time_check('processing', start_time, processing_time)
    tr_data = CustomDataset(tr_x, tr_y, tr_transform)

    tr_loader = DataLoader(tr_data, batch_size=Batch_size, shuffle=True)

    te_data = CustomDataset(te_x, te_y, te_transform)
    te_loader = DataLoader(te_data, batch_size=Batch_size, shuffle=False)

    loader_time = time.time()
    time_check('loading', processing_time, loader_time)

    model = models.resnet18()

    model.fc = nn.Linear(512, 4)
    model = model.cuda()

    # Summary Model
    summary(model, input_size=(3, Resolution, Resolution))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()

    for epoch in range(Epoch_num):
        train_loss = 0
        correct = 0
        total = 0
        for inputs, targets in tr_loader:
            inputs = inputs.cuda()
            targets = targets.cuda()

            outputs = model(inputs)
            # Ground True vs Prediction value
            # print(outputs)
            # print(targets)

            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        loss = train_loss / len(tr_loader)
        acc = 100 * correct / total

        if epoch % 5 == 0:
            print('Train - Epoch : %d, Loss : %.2f, Accuracy : %.2f' % (epoch, loss, acc))
            writer.add_scalar('Training loss', loss, epoch)
            writer.add_scalar('Accuracy', acc, epoch)

    train_time = time.time()
    time_check('train', loader_time, train_time)

    # https://tutorials.pytorch.kr/beginner/saving_loading_models.html
    torch.save(model, save_path + '/model.pt')

    # Evaluation
    model.eval()

    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(te_loader):
            inputs = inputs.cuda()
            targets = targets.cuda()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    loss = test_loss / len(te_loader)
    acc = 100 * correct / total

    print('Evaluation - Loss : %.2f, Accuracy : %.2f' % (loss, acc))

    test_time = time.time()
    time_check('test', train_time, test_time)
    writer.close()
