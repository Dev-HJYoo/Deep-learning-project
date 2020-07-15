# Import the neccessary libraries
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
import torchvision.models as models
import numpy as np
import time
from PIL import Image

# https://medium.com/@umerfarooq_26378/model-summary-in-pytorch-b5a1e4b64d25
# Add for summary function fo tensorflow.
from torchsummary import summary

# This script is to get image and label from folder
from mk_tensor import make_tensor

# It is to store model.
from torch.utils.tensorboard import SummaryWriter
import os


# https://data-panic.tistory.com/21
# Make custom dataset referring above URL.
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
        # Refer above URL
        x = Image.fromarray((self.x[item] * 255).astype(np.uint8))
        x = self.x_transform(x)

        # https://discuss.pytorch.org/t/runtimeerror-expected-object-of-scalar-type-long-but-got-scalar-type-float-when-using-crossentropyloss/30542
        y = self.y[item]

        return x, y


def time_check(name, start, end):
    print("{} time : {:0.3f}".format(name, end - start))

# Easily make 'requirement.txt'
# https://itinerant.tistory.com/100

def train():
    start_time = time.time()
    learning_rate = 0.001
    Epoch_num = 50
    Batch_size = 1024
    Resolution = 36
    save_path = 'save/'

    # https://tutorials.pytorch.kr/intermediate/tensorboard_tutorial.html
    # The direction for tensorboard
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

    # https://gldmg.tistory.com/43
    # This reference is to store and get .npy file

    # https://wikidocs.net/14304
    # Check if the file is located.
    train_path = 'data/pre_train'
    train_file = 'data/train.npy'

    test_path = 'data/pre_test'
    test_file = 'data/test.npy'

    if os.path.isfile(train_file):
        load = np.load(train_file, allow_pickle=True)
        tr_x, tr_y = load[0], load[1]
    else:
        tr_x, tr_y = make_tensor(train_path)
        np.save(train_file, np.array([tr_x, tr_y]))

    if os.path.isfile(test_file):
        load = np.load(test_file, allow_pickle=True)
        te_x, te_y = load[0], load[1]
    else:
        te_x, te_y = make_tensor(test_path)
        np.save(test_file, np.array([te_x, te_y]))

    processing_time = time.time()
    time_check('processing', start_time, processing_time)
    tr_data = CustomDataset(tr_x, tr_y, tr_transform)

    tr_loader = DataLoader(tr_data, batch_size=Batch_size, shuffle=True)

    te_data = CustomDataset(te_x, te_y, te_transform)
    te_loader = DataLoader(te_data, batch_size=Batch_size, shuffle=False)

    loader_time = time.time()
    time_check('loading', processing_time, loader_time)

    # model
    model = models.resnet18()

    model.fc = nn.Linear(512, 4)
    model = model.cuda()

    # Summary Model
    summary(model, input_size=(3, Resolution, Resolution))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Start training
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

if __name__ == '__main__':
    train()
