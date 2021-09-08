import torch
import torch.nn as nn
import torch.optim as opt
torch.set_printoptions(linewidth=120)
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import numpy as np

print(np.__version__)
print(torch.__version__)


def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size = 2, stride = 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size = 2, stride = 2)
        x = torch.flatten(x,start_dim = 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)

        return x



train_set = torchvision.datasets.FashionMNIST(root="./data",
                                                train = True,
                                                download=True,
                                                transform=transforms.ToTensor())
#
# train_loader = DataLoader(train_set,
#                             batch_size = 100,
#                             shuffle = True)
#
# tb = SummaryWriter()
# model = CNN()
# images, labels = next(iter(train_loader))
# grid = torchvision.utils.make_grid(images)
# tb.add_image("images", grid)
# tb.add_graph(model)
# tb.close()


device = ('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN().to(device)
train_loader = DataLoader(train_set,
                           batch_size = 100,
                           shuffle = True)
optimizer = opt.Adam(model.parameters(), lr= 0.01)
criterion = torch.nn.CrossEntropyLoss()


from itertools import product
parameters = dict(
    lr = [0.01, 0.001],
    batch_size = [32,64,128],
    shuffle = [True, False]
)


param_values = [v for v in parameters.values()]
print(param_values)

for lr, batch_size, shuffle in product(*param_values):
    print(lr, batch_size, shuffle)

for run_id, (lr, batch_size, shuffle) in enumerate(product(*param_values)):
    print("run id:", run_id + 1)
    model = CNN().to(device)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = shuffle)
    optimizer = opt.Adam(model.parameters(), lr = lr)
    criterion = torch.nn.CrossEntropyLoss()

    comment = f' batch_size = {batch_size} lr = {lr} shuffle = {shuffle}'
    tb = SummaryWriter(comment=comment)


    for epoch in range(3):
        total_loss = 0
        total_correct = 0
        for idx, (images, labels) in enumerate(train_loader):
            # grid = torchvision.utils.make_grid(images)
            # tb.add_image("input images", grid, global_step=idx)

            images, labels = images.to(device), labels.to(device)
            if idx == 0:
                tb.add_graph(model, images)

            preds = model(images)

            loss = criterion(preds, labels)
            total_loss += loss.item()
            total_correct += get_num_correct(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        tb.add_scalar("Loss", total_loss, epoch)
        tb.add_scalar("Correct", total_correct, epoch)
        tb.add_scalar("Accuracy", total_correct / len(train_set), epoch)

        for name, weight in model.named_parameters():
            tb.add_histogram(name, weight, epoch)
            tb.add_histogram(f'{name}.grad', weight.grad, epoch)

        print("batch_size:", batch_size, "lr:", lr, "shuffle:", shuffle)
        print("epoch:", epoch, "total_correct:", total_correct, "loss:", total_loss)
    print("__________________________________________________________")

    tb.add_hparams(
            {"lr": lr, "bsize": batch_size, "shuffle":shuffle},
            {"accuracy": total_correct/len(train_set), "loss": total_loss}
        )

tb.close()
