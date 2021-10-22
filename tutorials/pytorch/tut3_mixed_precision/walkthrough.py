#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

# Half and mixed precision in PopTorch

# This tutorial shows how to use half and mixed precision in PopTorch with the
# example task of training a simple CNN model on a single
# Graphcore IPU (Mk1 or Mk2).

# Requirements:
#   - an installed Poplar SDK. See the Getting Started guide for your IPU
#     hardware for details of how to install the SDK;
#   - Other Python modules: `pip install -r requirements.txt`

# Import the packages
import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms

import poptorch
import argparse
from tqdm import tqdm


# Build the model
class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 5, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(5, 12, 5)
        self.norm = nn.GroupNorm(3, 12)
        self.fc1 = nn.Linear(41772, 100)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(100, 10)
        self.log_softmax = nn.LogSoftmax(dim=0)
        self.loss = nn.NLLLoss()

    def forward(self, x, labels=None):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.norm(self.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.log_softmax(self.fc2(x))
        # The model is responsible for the calculation
        # of the loss when using an IPU. We do it this way:
        if self.training:
            return x, self.loss(x, labels)
        return x

model = CustomModel()

parser = argparse.ArgumentParser()
parser.add_argument('--model-half', dest='model_half', action='store_true', help='Cast the model parameters to FP16')
parser.add_argument('--data-half', dest='data_half', action='store_true', help='Cast the data to FP16')
parser.add_argument('--optimizer-half', dest='optimizer_half', action='store_true', help='Cast the accumulation type of the optimiser to FP16')
parser.add_argument('--stochastic-rounding', dest='stochastic_rounding', action='store_true', help='Use stochasting rounding')
parser.add_argument('--partials-half', dest='partials_half', action='store_true', help='Set partials data type to FP16')
args = parser.parse_args()

# Casting a model's parameters
if args.model_half:
    model = model.half()

# Prepare the data
if args.data_half:
    transform = transforms.Compose([transforms.Resize(128),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    transforms.ConvertImageDtype(torch.half)])
else:
    transform = transforms.Compose([transforms.Resize(128),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

train_dataset = torchvision.datasets.FashionMNIST("~/.torch/datasets",
                                                  transform=transform,
                                                  download=True,
                                                  train=True)
test_dataset = torchvision.datasets.FashionMNIST("~/.torch/datasets",
                                                 transform=transform,
                                                 download=True,
                                                 train=False)

# Optimizer and loss scaling
if args.optimizer_half:
    optimizer = poptorch.optim.AdamW(model.parameters(),
                                     lr=0.001,
                                     loss_scaling=1024,
                                     accum_type=torch.float16)
else:
    optimizer = poptorch.optim.AdamW(model.parameters(),
                                     lr=0.001,
                                     accum_type=torch.float32)


# Set PopTorch's options
opts = poptorch.Options()

# Stochastic rounding
if args.stochastic_rounding:
    opts.Precision.enableStochasticRounding(True)
# Partials data type
if args.partials_half:
    opts.Precision.setPartialsType(torch.half)
else:
    opts.Precision.setPartialsType(torch.float)

# Train the model
train_dataloader = poptorch.DataLoader(opts,
                                       train_dataset,
                                       batch_size=12,
                                       shuffle=True,
                                       num_workers=40)
model.train()  # Switch the model to training mode
# Models are initialised in training mode by default, so the line above will
# have no effect. Its purpose is to show how the mode can be set explicitly.

poptorch_model = poptorch.trainingModel(model,
                                        options=opts,
                                        optimizer=optimizer)

epochs = 10
for epoch in tqdm(range(epochs), desc="epochs"):
    total_loss = 0.0
    for data, labels in tqdm(train_dataloader, desc="batches", leave=False):
        output, loss = poptorch_model(data, labels)
        total_loss += loss

# Evaluate the model
model.eval()  # Switch the model to inference mode
poptorch_model_inf = poptorch.inferenceModel(model, options=opts)
test_dataloader = poptorch.DataLoader(opts,
                                      test_dataset,
                                      batch_size=32,
                                      num_workers=40)

predictions, labels = [], []
for data, label in test_dataloader:
    predictions += poptorch_model_inf(data).data.float().max(dim=1).indices
    labels += label

print(f"""Eval accuracy on IPU: {100 *
                (1 - torch.count_nonzero(torch.sub(torch.tensor(labels),
                torch.tensor(predictions))) / len(labels)):.2f}%""")
