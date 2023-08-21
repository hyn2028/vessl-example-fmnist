import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import vessl

import random
from random import choice
from argparse import ArgumentParser
from pathlib import Path

from simple_cnn import SimpleCNN
from run_lib import train, test, prepare_data, figure_to_array

argparser = ArgumentParser()
argparser.add_argument("dataset_root", type=Path)
argparser.add_argument("output_root", type=Path)
argparser.add_argument("--cls_hidden", type=int, default=128)
argparser.add_argument("--dropout", type=float, default=0.5)
argparser.add_argument("--batch_size", type=int, default=128)
argparser.add_argument("--epochs", type=int, default=30)
argparser.add_argument("--lr", type=float, default=1e-3)
argparser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
argparser.add_argument("--seed", type=int, default=42)
args = argparser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


# tensorboard logging integration with VESSL
# hyperparameters are automatically logged to VESSL
vessl.init(tensorboard=True, hp=args)
writer = SummaryWriter(args.output_root / "logs")


training_data, test_data, train_dataloader, test_dataloader, class_names = prepare_data(root=args.dataset_root, batch_size=args.batch_size)


for X, y in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break


# get deivce for traning
print("Using {} device".format(args.device), end="")
if torch.cuda.is_available():
     print(" ({})".format(torch.cuda.get_device_name(0)))


# define a model
model = SimpleCNN(cls_hidden=args.cls_hidden, dropout=args.dropout).to(args.device)
print(model)


# define loss function and optimizer
loss_fn = nn.CrossEntropyLoss() # softmax + cross_entropy
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


# train and test
best_acc = 0
best_loss = 0
best_step = 0
for t in range(args.epochs):
    step = t * len(train_dataloader)
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer, args, writer, step)

    step = (t + 1) * len(train_dataloader)
    test_loss, test_acc = test(test_dataloader, model, loss_fn, args, writer, step)

    if test_acc > best_acc:
        best_acc = test_acc
        best_loss = test_loss
        best_step = step
        torch.save(model.state_dict(), args.output_root / "best_model.pth")
    
    vessl.progress((t + 1) / args.epochs)
print("Done!")


# reload best model
model.load_state_dict(torch.load(args.output_root / "best_model.pth"))

writer.add_scalar("best/loss", best_loss, best_step)
writer.add_scalar("best/acc", best_acc, best_step)


# plot weights
w1 = model.conv1.get_parameter("weight").cpu().detach()
fig = plt.figure()
for i in range(32):
    ax = plt.subplot(4,8,i+1)
    plt.matshow(w1[i, 0], fignum=False)
fig = figure_to_array(fig)
writer.add_image("best/conv1", fig, global_step=best_step)

w2 = model.conv2.get_parameter("weight").cpu().detach()
fig = plt.figure()
for i in range(64):
    ax = plt.subplot(8,8,i+1)
    plt.matshow(w2[i, 0], fignum=False)
fig = figure_to_array(fig)
writer.add_image("best/conv2", fig, global_step=best_step)

writer.close()
