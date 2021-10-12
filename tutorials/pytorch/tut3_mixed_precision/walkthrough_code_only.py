# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import poptorch
from tqdm.auto import tqdm

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

# Cast the model parameters to FP16
model_half = True

# Cast the data to FP16
data_half = True

# Cast the accumulation of gradients values types of the optimiser to FP16
optimizer_half = True

# Use stochasting rounding
stochastic_rounding = True

# Set partials data type to FP16
partials_half = False

model = CustomModel()

if model_half:
    model = model.half()

transform_list = [transforms.Resize(128),
                  transforms.ToTensor(),
                  transforms.Normalize((0.5,), (0.5,))]
if data_half:
    transform_list.append(transforms.ConvertImageDtype(torch.half))

transform = transforms.Compose(transform_list)

train_dataset = torchvision.datasets.FashionMNIST("./datasets/",
                                                  transform=transform,
                                                  download=True,
                                                  train=True)
test_dataset = torchvision.datasets.FashionMNIST("./datasets/",
                                                 transform=transform,
                                                 download=True,
                                                 train=False)

accum, loss_scaling = \
    (torch.float16, 1024) if optimizer_half else (torch.float32, None)

optimizer = poptorch.optim.AdamW(params=model.parameters(),
                                 lr=0.001,
                                 accum_type=accum,
                                 loss_scaling=loss_scaling)

opts = poptorch.Options()

if stochastic_rounding:
    opts.Precision.enableStochasticRounding(True)

if partials_half:
    opts.Precision.setPartialsType(torch.half)
else:
    opts.Precision.setPartialsType(torch.float)

train_dataloader = poptorch.DataLoader(opts,
                                       train_dataset,
                                       batch_size=12,
                                       shuffle=True,
                                       num_workers=40)

model.train()
poptorch_model = poptorch.trainingModel(model,
                                        options=opts,
                                        optimizer=optimizer)

epochs = 10
for epoch in tqdm(range(epochs), desc="epochs"):
    total_loss = 0.0
    for data, labels in tqdm(train_dataloader, desc="batches", leave=False):
        output, loss = poptorch_model(data, labels)
        total_loss += loss

poptorch_model.detachFromDevice()

model.eval()
poptorch_model_inf = poptorch.inferenceModel(model, options=opts)
test_dataloader = poptorch.DataLoader(opts,
                                      test_dataset,
                                      batch_size=32,
                                      num_workers=40)

predictions, labels = [], []
for data, label in test_dataloader:
    predictions += poptorch_model_inf(data).data.float().max(dim=1).indices
    labels += label

poptorch_model_inf.detachFromDevice()

print(f"""Eval accuracy on IPU: {100 *
                (1 - torch.count_nonzero(torch.sub(torch.tensor(labels),
                torch.tensor(predictions))) / len(labels)):.2f}%""")

class Model(torch.nn.Module):
    def forward(self, x, y):
        return x + y

native_model = Model()

float16_tensor = torch.tensor([1.0], dtype=torch.float16)
float32_tensor = torch.tensor([1.0], dtype=torch.float32)

assert native_model(float32_tensor, float16_tensor).dtype == torch.float32

opts = poptorch.Options()

poptorch_model = poptorch.inferenceModel(native_model, opts)
assert poptorch_model(float32_tensor, float16_tensor).dtype == torch.float16

opts.Precision.halfFloatCasting(
    poptorch.HalfFloatCastingBehavior.HalfUpcastToFloat)

poptorch_model = poptorch.inferenceModel(native_model, opts)
assert poptorch_model(float32_tensor, float16_tensor).dtype == torch.float32

poptorch_model.detachFromDevice()
