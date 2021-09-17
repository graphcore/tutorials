# Introduction to PopTorch - running a simple model

This tutorial covers the basics of model making in PyTorch,
using `torch.nn.Module`, and the specific methods to
convert a PyTorch model to a PopTorch model so that it can
be run on a Graphcore IPU.

Requirements:
   - an installed Poplar SDK. See the Getting Started guide
     for your IPU hardware for details of how to install
     the SDK;
   - Python packages with `pip install -r requirements.txt`

# Table of Contents
- [What is PopTorch?](#What-is-PopTorch?)
- [Getting started: training a model on the IPU](#Getting-started:-training-a-model-on-the-IPU)
    - [Import the packages](#Import-the-packages)
    - [Load the data](#Load-the-data)
        - [PopTorch DataLoader](#PopTorch-DataLoader)
    - [Build the model](#Build-the-model)
    - [Prepare training for IPUs](#Prepare-training-for-IPUs)
    - [Train the model](#Train-the-model)
      - [Training loop](#Training-loop)
      - [Use the same IPU for training and inference](#Use-the-same-IPU-for-training-and-inference)
      - [Save the trained model](#Save-the-trained-model)
    - [Evaluate the model](#Evaluate-the-model)
- [Doing more with `poptorch.Options`](#Doing-more-with-poptorch.Options)
    - [`deviceIterations`](#deviceIterations)
    - [`replicationFactor`](#replicationFactor)
    - [`randomSeed`](#randomSeed)
    - [`useIpuModel`](#useIpuModel)
    - [How to set the options](#How-to-set-the-options)
- [Going further](#Going-further)

## What is PopTorch?
PopTorch is a set of extensions for PyTorch to enable
PyTorch models to run on Graphcore's IPU hardware.

PopTorch supports both inference and training. To run a
model on the IPU you wrap your existing PyTorch model in
either a PopTorch inference wrapper or a PopTorch training
wrapper. You can provide further annotations to partition
the model across multiple IPUs.

You can wrap individual layers in an IPU helper to
designate which IPU they should go on. Using your
annotations, PopTorch will use [PopART]
(https://docs.graphcore.ai/projects/popart-user-guide)
to parallelise the model over the given number of IPUs.
Additional parallelism can be expressed via a replication
factor which enables you to data-parallelise the model over
more IPUs.

Under the hood PopTorch uses [TorchScript](https://pytorch.org/docs/stable/jit.html),
an intermediate representation (IR) of a PyTorch model,
using the `torch.jit.# trace` API. That means it inherits
the constraints of that API. These include:

- Inputs must be Torch tensors or tuples/lists containing Torch tensors;
- None can be used as a default value for a parameter but cannot be
expliticly passed as an input value;
- Hooks and `.grad` cannot be used to inspect weights and gradients;
- `torch.jit.trace` cannot handle control flow or shape
variations within the model. That is, the inputs passed at
run-time cannot vary the control flow of the model or the
shapes/sizes of results.

To learn more about TorchScript and JIT, you can go through
this [tutorial](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html).

PopTorch has been designed to require few manual
alterations to your models in order to run them on IPU.
However, it does have some differences from native PyTorch
execution. Also, not all PyTorch operations have been
implemented by the backend yet. You can find the list of
supported operations [here](https://docs.graphcore.ai/projects/poptorch-user-guide/en/latest supported_ops.html).

![Software stack](static/stack.jpg)

# Getting started: training a model on the IPU
We will do the following steps in order:
1. Load the Fashion-MNIST dataset using `torchvision.datasets` and `poptorch.
DataLoader`
2. Define a deep CNN  and a loss function using the `torch` API
3. Train the model on an IPU using `poptorch.trainingModel`
4. Evaluate the model on the IPU

### Import the packages
PopTorch is a separate package from PyTorch, and available
in Graphcore's Poplar SDK. Both must thus be imported:


```python
import torch
import poptorch
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
```

Under the hood, PopTorch uses Graphcore's high-performance
machine learning framework PopART. It is therefore necessary
to enable PopART and Poplar in your environment.

>**NOTE**:
>If you forget PopART, you will encounter the error
>`ImportError: libpopart.so: cannot open shared object file: No such file or
>directory` when importing `poptorch`.
>If the error message says something like `libpopart_compiler.so: undefined
>symbol: _ZN6popart7Session3runERNS_7IStepIOE`, it most likely means the
>versions of PopART and PopTorch do not match, for example by enabling PopART
>with a previous SDK release's `enable.sh` script. Make sure to not mix SDK's
>artifacts.

### Load the data
We will use the Fashion-MNIST dataset made available by the package
`torchivsion`. This dataset, from [Zalando](https://github.com/zalandoresearch/fashion-mnist),
can be used as a more challenging replacement to the well-known MNIST dataset.

The dataset consists of 28x28 grayscale images and labels of range \[0, 9]
from 10 classes: T-shirt, trouser, pullover, dress, coat, sandal, shirt,
sneaker, bag and ankle boot.

In order for the images to be usable by PyTorch, we have to convert them to
`torch.Tensor` objects. Also, data normalisation improves overall
performance. We will apply both operations, conversion and normalisation, to
the datasets using `torchvision.transforms` and feed these ops to
`torchvision.datasets`:


```python
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,), (0.5,))])

train_dataset = torchvision.datasets.FashionMNIST("./datasets/", transform=transform, download=True, train=True)
test_dataset = torchvision.datasets.FashionMNIST("./datasets/", transform=transform, download=True, train=False)
classes = ("T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot")

```

    /home/adamw/adam_env/lib/python3.6/site-packages/torchvision/datasets/mnist.py:498: UserWarning: The given NumPy array is not writeable, and PyTorch does not support non-writeable tensors. This means you can write to the underlying (supposedly non-writeable) NumPy array using the tensor. You may want to copy the array to protect its data or make it writeable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at  /pytorch/torch/csrc/utils/tensor_numpy.cpp:180.)
      return torch.from_numpy(parsed.astype(m[2], copy=False)).view(*s)


With the following method, we can visualise a sample of these images and
their associated labels:


```python
plt.figure(figsize=(30, 15))
for i, (image, label) in enumerate(train_dataset):
    if i == 15:
        break
    image = (image / 2 + .5).numpy()  # reverse transformation
    ax = plt.subplot(5, 5, i + 1)
    ax.set_title(classes[label])
    plt.imshow(image[0])
```


    
![png](walkthrough_syntax_keras3_executed_files/walkthrough_syntax_keras3_executed_22_0.png)
    


![png](static/from_0_to_1_10_0.png)

##### PopTorch DataLoader
We can feed batches of data into a PyTorch model by simply passing the input
tensors. However, this is unlikely to be the most efficient way and can
result in data loading being a bottleneck to the model, slowing down the
training process. In order to make data loading easier and more efficient,
there's the [`torch.utils.data.DataLoader`](https://pytorch.org/docs/stable/data.html)
class, which is an iterable over a dataset and which can handle parallel data
loading, a sampling strategy, shuffling, etc.

PopTorch offers an extension of this class with its [`poptorch.DataLoader`]
(https://docs.graphcore.ai/projects/poptorch-user-guide/en/latest/batching.html#poptorch-dataloader)
class, specialised for the way the underlying PopART framework handles
batching of data. We will use this class later in the tutorial, as soon as we
have a model ready for training.

### Build the model
We will build a simple CNN model for a classification task. To do so, we can
simply use PyTorch's API, including `torch.nn.Module`. The difference from
what we're used to with pure PyTorch is the _loss computation_, which has to
be part of the `forward` function. This is to ensure the loss is computed on
the IPU and not on the CPU, and to give us as much flexibility as possible
when designing more complex loss functions.


```python

class ClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 5, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(5, 12, 5)
        self.norm = nn.GroupNorm(3, 12)
        self.fc1 = nn.Linear(972, 100)
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

model = ClassificationModel()

```

**NOTE**: `self.training` is inherited from `torch.nn.Module` which
initialises its value to `True`. Use `model.eval()` to set it to `False` and
`model.train()` to switch it back to `True`.

### Prepare training for IPUs
The compilation and execution on the IPU can be controlled using `poptorch.
Options`. These options are used by PopTorch's wrappers such as `poptorch.
DataLoader` and `poptorch.trainingModel`.


```python
opts = poptorch.Options()

train_dataloader = poptorch.DataLoader(opts, train_dataset, batch_size=16, shuffle=True, num_workers=20)

```

### Train the model
We will need another component in order to train our model: an optimiser.
Its role is to apply the computed gradients to the model's weights to optimize
(usually, minimize) the loss function using a specific algorithm. PopTorch
currently provides classes which inherit from multiple native PyTorch optimisation
functions: SGD, Adam, AdamW, LAMB and RMSprop. These optimisers provide several
advantages over native PyTorch versions. They embed constant attributes to save
performance and memory, and allow you to specify additional parameters such as
loss/velocity scaling.

We will use SGD (https://docs.graphcore.ai/projects/poptorch-user-guide/en/latest/reference.html#poptorch.optim.SGD)
as it's a very popular algorithm and is appropriate for this classification task.


```python
optimizer = poptorch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

```

    [03:32:29.183] [poptorch::python] [warning] Default SGD implementation has changed to the more stable but more memory intensive separate variant. To suppress set use_combined_accum=False with poptorch.optim.SGD; to restore old behaviour, set use_combined_accum=True


We now introduce the `poptorch.trainingModel` wrapper, which will handle the
training. It takes an instance of a `torch.nn.Module`, such as our custom
model, an instance of `poptorch.Options` which we have instantiated
previously, and an optimizer. This wrapper will trigger the compilation of
our model, using TorchScript, and manage its translation to a program the
IPU can run. Let's use it.


```python
poptorch_model = poptorch.trainingModel(model, options=opts, optimizer=optimizer)

```

#### Training loop
Looping through the training data, running the forward and backward passes,
and updating the weights constitute the process we refer to as the "training loop".
Graphcore's Poplar system uses several optimisations to accelerate the
training loop. Central to this is the desire to minimise interactions between
the device (the IPU) and the host (the CPU), allowing the training loop to
run on the device independently from the host. To achieve that virtual
independence, Poplar creates a _static_ computational graph and data streams
which are loaded to the IPU, and then signals the IPU to get started until
there's no data left or until the host sends a signal to stop the loop.

![High-level overview of what happens](static/loop.jpg)

The compilation, which transforms our PyTorch model into a computational
graph and our dataloader into data streams, happens at the first call of a
`poptorch.trainingModel`. The IPUs to which the graph will be uploaded are
selected automatically during this first call, by default. The training loop can
then start.

Once the loop has started, Poplar's main task is to feed the data into the
streams and to signal when we are done with the loop. The last step will then
be to copy the final graph, meaning the model, back to the CPU - a step that
PopTorch manages itself.


```python
epochs = 30
for epoch in tqdm(range(epochs), desc="epochs"):
    total_loss = 0.0
    for data, labels in tqdm(train_dataloader, desc="batches", leave=False):
        output, loss = poptorch_model(data, labels)
        total_loss += loss

```

    epochs:   0%|          | 0/30 [00:00<?, ?it/s]
    batches:   0%|          | 0/3750 [00:00<?, ?it/s][A
    
    Graph compilation:   0%|          | 0/100 [00:00<?][A[A
    
    Graph compilation:   3%|▎         | 3/100 [00:00<00:03][A[A
    
    Graph compilation:   6%|▌         | 6/100 [00:08<02:44][A[A
    
    Graph compilation:   8%|▊         | 8/100 [00:11<02:24][A[A
    
    Graph compilation:  17%|█▋        | 17/100 [00:11<00:40][A[A
    
    Graph compilation:  23%|██▎       | 23/100 [00:11<00:23][A[A
    
    Graph compilation:  26%|██▌       | 26/100 [00:22<01:14][A[A
    
    Graph compilation:  27%|██▋       | 27/100 [00:22<01:09][A[A
    
    Graph compilation:  29%|██▉       | 29/100 [00:22<00:53][A[A
    
    Graph compilation:  31%|███       | 31/100 [00:24<00:50][A[A
    
    Graph compilation:  33%|███▎      | 33/100 [00:25<00:46][A[A
    
    Graph compilation:  42%|████▏     | 42/100 [00:25<00:16][A[A
    
    Graph compilation:  47%|████▋     | 47/100 [00:25<00:10][A[A
    
    Graph compilation:  50%|█████     | 50/100 [00:25<00:08][A[A
    
    Graph compilation:  53%|█████▎    | 53/100 [00:26<00:09][A[A
    
    Graph compilation:  64%|██████▍   | 64/100 [00:28<00:06][A[A
    
    Graph compilation:  67%|██████▋   | 67/100 [00:28<00:05][A[A
    
    Graph compilation:  73%|███████▎  | 73/100 [00:29<00:03][A[A
    
    Graph compilation:  76%|███████▌  | 76/100 [00:29<00:02][A[A
    
    Graph compilation:  82%|████████▏ | 82/100 [00:29<00:01][A[A
    
    Graph compilation:  84%|████████▍ | 84/100 [00:30<00:02][A[A
    
    Graph compilation:  87%|████████▋ | 87/100 [00:31<00:01][A[A
    
    Graph compilation:  89%|████████▉ | 89/100 [00:33<00:03][A[A
    
    Graph compilation:  90%|█████████ | 90/100 [00:33<00:03][A[A
    
    Graph compilation:  91%|█████████ | 91/100 [00:33<00:02][A[A
    
    Graph compilation:  92%|█████████▏| 92/100 [00:34<00:02][A[A
    
    Graph compilation:  93%|█████████▎| 93/100 [00:34<00:02][A[A
    
    Graph compilation: 100%|██████████| 100/100 [00:39<00:00][A[A
    
    batches:   0%|          | 1/3750 [00:43<44:47:51, 43.02s/it][A
    batches:   2%|▏         | 75/3750 [00:43<24:46,  2.47it/s]  [A
    batches:   4%|▍         | 156/3750 [00:43<09:32,  6.27it/s][A
    batches:   6%|▋         | 239/3750 [00:43<04:58, 11.77it/s][A
    batches:   9%|▊         | 321/3750 [00:43<02:56, 19.41it/s][A
    batches:  11%|█         | 404/3750 [00:43<01:50, 30.20it/s][A
    batches:  13%|█▎        | 487/3750 [00:43<01:12, 45.13it/s][A
    batches:  15%|█▌        | 569/3750 [00:43<00:48, 65.14it/s][A
    batches:  17%|█▋        | 651/3750 [00:43<00:33, 92.10it/s][A
    batches:  20%|█▉        | 734/3750 [00:43<00:23, 127.88it/s][A
    batches:  22%|██▏       | 816/3750 [00:44<00:16, 172.68it/s][A
    batches:  24%|██▍       | 898/3750 [00:44<00:12, 227.46it/s][A
    batches:  26%|██▌       | 980/3750 [00:44<00:09, 290.86it/s][A
    batches:  28%|██▊       | 1063/3750 [00:44<00:07, 362.23it/s][A
    batches:  31%|███       | 1145/3750 [00:44<00:05, 435.35it/s][A
    batches:  33%|███▎      | 1227/3750 [00:44<00:05, 503.16it/s][A
    batches:  35%|███▍      | 1310/3750 [00:44<00:04, 570.59it/s][A
    batches:  37%|███▋      | 1394/3750 [00:44<00:03, 630.67it/s][A
    batches:  39%|███▉      | 1478/3750 [00:44<00:03, 680.39it/s][A
    batches:  42%|████▏     | 1561/3750 [00:44<00:03, 718.45it/s][A
    batches:  44%|████▍     | 1644/3750 [00:45<00:02, 745.89it/s][A
    batches:  46%|████▌     | 1727/3750 [00:45<00:02, 768.80it/s][A
    batches:  48%|████▊     | 1810/3750 [00:45<00:03, 601.21it/s][A
    batches:  50%|█████     | 1892/3750 [00:45<00:02, 652.41it/s][A
    batches:  53%|█████▎    | 1975/3750 [00:45<00:02, 695.39it/s][A
    batches:  55%|█████▍    | 2058/3750 [00:45<00:02, 729.30it/s][A
    batches:  57%|█████▋    | 2141/3750 [00:45<00:02, 755.39it/s][A
    batches:  59%|█████▉    | 2221/3750 [00:45<00:02, 755.29it/s][A
    batches:  61%|██████▏   | 2303/3750 [00:45<00:01, 772.46it/s][A
    batches:  64%|██████▎   | 2386/3750 [00:46<00:01, 786.99it/s][A
    batches:  66%|██████▌   | 2468/3750 [00:46<00:01, 795.02it/s][A
    batches:  68%|██████▊   | 2549/3750 [00:46<00:01, 790.91it/s][A
    batches:  70%|███████   | 2632/3750 [00:46<00:01, 801.09it/s][A
    batches:  72%|███████▏  | 2713/3750 [00:46<00:01, 793.81it/s][A
    batches:  75%|███████▍  | 2795/3750 [00:46<00:01, 800.74it/s][A
    batches:  77%|███████▋  | 2876/3750 [00:46<00:01, 796.49it/s][A
    batches:  79%|███████▉  | 2956/3750 [00:46<00:00, 796.48it/s][A
    batches:  81%|████████  | 3039/3750 [00:46<00:00, 803.45it/s][A
    batches:  83%|████████▎ | 3120/3750 [00:46<00:00, 793.61it/s][A
    batches:  85%|████████▌ | 3202/3750 [00:47<00:00, 800.92it/s][A
    batches:  88%|████████▊ | 3283/3750 [00:47<00:00, 565.00it/s][A
    batches:  89%|████████▉ | 3350/3750 [00:47<00:00, 470.29it/s][A
    batches:  91%|█████████ | 3406/3750 [00:47<00:00, 375.70it/s][A
    batches:  92%|█████████▏| 3452/3750 [00:47<00:00, 332.59it/s][A
    batches:  93%|█████████▎| 3492/3750 [00:48<00:00, 319.53it/s][A
    batches:  95%|█████████▌| 3574/3750 [00:48<00:00, 417.24it/s][A
    batches:  97%|█████████▋| 3656/3750 [00:48<00:00, 505.03it/s][A
    batches: 100%|█████████▉| 3739/3750 [00:48<00:00, 581.22it/s][A
    epochs:   3%|▎         | 1/30 [00:48<23:24, 48.43s/it]       [A
    batches:   0%|          | 0/3750 [00:00<?, ?it/s][A
    batches:   2%|▏         | 62/3750 [00:00<00:05, 618.21it/s][A
    batches:   4%|▎         | 138/3750 [00:00<00:05, 699.59it/s][A
    batches:   6%|▌         | 214/3750 [00:00<00:04, 723.10it/s][A
    batches:   8%|▊         | 289/3750 [00:00<00:04, 731.17it/s][A
    batches:  10%|▉         | 365/3750 [00:00<00:04, 740.49it/s][A
    batches:  12%|█▏        | 441/3750 [00:00<00:04, 744.51it/s][A
    batches:  14%|█▍        | 516/3750 [00:00<00:04, 743.55it/s][A
    batches:  16%|█▌        | 593/3750 [00:00<00:04, 749.07it/s][A
    batches:  18%|█▊        | 670/3750 [00:00<00:04, 753.10it/s][A
    batches:  20%|█▉        | 747/3750 [00:01<00:03, 756.55it/s][A
    batches:  22%|██▏       | 824/3750 [00:01<00:03, 758.26it/s][A
    batches:  24%|██▍       | 901/3750 [00:01<00:03, 758.61it/s][A
    batches:  26%|██▌       | 978/3750 [00:01<00:03, 759.99it/s][A
    batches:  28%|██▊       | 1055/3750 [00:01<00:03, 760.11it/s][A
    batches:  30%|███       | 1132/3750 [00:01<00:03, 760.46it/s][A
    batches:  32%|███▏      | 1209/3750 [00:01<00:03, 761.95it/s][A
    batches:  34%|███▍      | 1286/3750 [00:01<00:03, 760.00it/s][A
    batches:  36%|███▋      | 1363/3750 [00:01<00:03, 760.79it/s][A
    batches:  38%|███▊      | 1440/3750 [00:01<00:03, 762.82it/s][A
    batches:  40%|████      | 1517/3750 [00:02<00:02, 764.06it/s][A
    batches:  43%|████▎     | 1594/3750 [00:02<00:02, 753.45it/s][A
    batches:  45%|████▍     | 1671/3750 [00:02<00:02, 756.13it/s][A
    batches:  47%|████▋     | 1748/3750 [00:02<00:02, 760.16it/s][A
    batches:  49%|████▊     | 1825/3750 [00:02<00:02, 723.75it/s][A
    batches:  51%|█████     | 1898/3750 [00:02<00:02, 705.24it/s][A
    batches:  53%|█████▎    | 1969/3750 [00:02<00:02, 691.96it/s][A
    batches:  54%|█████▍    | 2040/3750 [00:02<00:02, 696.94it/s][A
    batches:  56%|█████▋    | 2110/3750 [00:02<00:02, 697.31it/s][A
    batches:  58%|█████▊    | 2185/3750 [00:02<00:02, 710.32it/s][A
    batches:  60%|██████    | 2260/3750 [00:03<00:02, 721.67it/s][A
    batches:  62%|██████▏   | 2336/3750 [00:03<00:01, 730.78it/s][A
    batches:  64%|██████▍   | 2412/3750 [00:03<00:01, 737.05it/s][A
    batches:  66%|██████▋   | 2489/3750 [00:03<00:01, 745.29it/s][A
    batches:  68%|██████▊   | 2565/3750 [00:03<00:01, 746.82it/s][A
    batches:  70%|███████   | 2640/3750 [00:03<00:01, 735.02it/s][A
    batches:  72%|███████▏  | 2714/3750 [00:03<00:01, 730.39it/s][A
    batches:  74%|███████▍  | 2788/3750 [00:03<00:01, 707.26it/s][A
    batches:  76%|███████▋  | 2862/3750 [00:03<00:01, 714.02it/s][A
    batches:  78%|███████▊  | 2939/3750 [00:03<00:01, 728.39it/s][A
    batches:  80%|████████  | 3016/3750 [00:04<00:00, 738.66it/s][A
    batches:  82%|████████▏ | 3093/3750 [00:04<00:00, 745.58it/s][A
    batches:  85%|████████▍ | 3169/3750 [00:04<00:00, 748.62it/s][A
    batches:  87%|████████▋ | 3244/3750 [00:04<00:00, 696.43it/s][A
    batches:  88%|████████▊ | 3315/3750 [00:04<00:00, 678.61it/s][A
    batches:  90%|█████████ | 3389/3750 [00:04<00:00, 694.37it/s][A
    batches:  92%|█████████▏| 3464/3750 [00:04<00:00, 707.55it/s][A
    batches:  94%|█████████▍| 3539/3750 [00:04<00:00, 717.59it/s][A
    batches:  96%|█████████▋| 3615/3750 [00:04<00:00, 728.64it/s][A
    batches:  98%|█████████▊| 3690/3750 [00:05<00:00, 734.15it/s][A
    epochs:   7%|▋         | 2/30 [00:53<10:42, 22.96s/it]       [A
    batches:   0%|          | 0/3750 [00:00<?, ?it/s][A
    batches:   1%|          | 44/3750 [00:00<00:08, 435.18it/s][A
    batches:   3%|▎         | 117/3750 [00:00<00:06, 605.07it/s][A
    batches:   5%|▌         | 193/3750 [00:00<00:05, 673.99it/s][A
    batches:   7%|▋         | 265/3750 [00:00<00:05, 691.34it/s][A
    batches:   9%|▉         | 341/3750 [00:00<00:04, 714.91it/s][A
    batches:  11%|█         | 414/3750 [00:00<00:04, 718.85it/s][A
    batches:  13%|█▎        | 486/3750 [00:00<00:04, 713.72it/s][A
    batches:  15%|█▍        | 558/3750 [00:00<00:04, 710.34it/s][A
    batches:  17%|█▋        | 631/3750 [00:00<00:04, 714.95it/s][A
    batches:  19%|█▉        | 705/3750 [00:01<00:04, 720.49it/s][A
    batches:  21%|██        | 780/3750 [00:01<00:04, 725.65it/s][A
    batches:  23%|██▎       | 853/3750 [00:01<00:03, 725.10it/s][A
    batches:  25%|██▍       | 926/3750 [00:01<00:04, 703.58it/s][A
    batches:  27%|██▋       | 1000/3750 [00:01<00:03, 711.97it/s][A
    batches:  29%|██▊       | 1075/3750 [00:01<00:03, 721.23it/s][A
    batches:  31%|███       | 1151/3750 [00:01<00:03, 730.83it/s][A
    batches:  33%|███▎      | 1227/3750 [00:01<00:03, 736.89it/s][A
    batches:  35%|███▍      | 1303/3750 [00:01<00:03, 741.58it/s][A
    batches:  37%|███▋      | 1378/3750 [00:01<00:03, 743.80it/s][A
    batches:  39%|███▊      | 1453/3750 [00:02<00:03, 745.43it/s][A
    batches:  41%|████      | 1529/3750 [00:02<00:02, 746.72it/s][A
    batches:  43%|████▎     | 1604/3750 [00:02<00:02, 747.65it/s][A
    batches:  45%|████▍     | 1680/3750 [00:02<00:02, 749.60it/s][A
    batches:  47%|████▋     | 1756/3750 [00:02<00:02, 749.92it/s][A
    batches:  49%|████▉     | 1831/3750 [00:02<00:02, 749.52it/s][A
    batches:  51%|█████     | 1906/3750 [00:02<00:02, 745.67it/s][A
    batches:  53%|█████▎    | 1981/3750 [00:02<00:02, 737.91it/s][A
    batches:  55%|█████▍    | 2056/3750 [00:02<00:02, 740.80it/s][A
    batches:  57%|█████▋    | 2131/3750 [00:02<00:02, 734.50it/s][A
    batches:  59%|█████▉    | 2207/3750 [00:03<00:02, 739.31it/s][A
    batches:  61%|██████    | 2281/3750 [00:03<00:02, 726.85it/s][A
    batches:  63%|██████▎   | 2355/3750 [00:03<00:01, 730.09it/s][A
    batches:  65%|██████▍   | 2430/3750 [00:03<00:01, 735.23it/s][A
    batches:  67%|██████▋   | 2505/3750 [00:03<00:01, 737.87it/s][A
    batches:  69%|██████▉   | 2580/3750 [00:03<00:01, 739.85it/s][A
    batches:  71%|███████   | 2655/3750 [00:03<00:01, 742.05it/s][A
    batches:  73%|███████▎  | 2730/3750 [00:03<00:01, 743.53it/s][A
    batches:  75%|███████▍  | 2806/3750 [00:03<00:01, 744.56it/s][A
    batches:  77%|███████▋  | 2882/3750 [00:03<00:01, 746.53it/s][A
    batches:  79%|███████▉  | 2957/3750 [00:04<00:01, 696.48it/s][A
    batches:  81%|████████  | 3028/3750 [00:04<00:01, 695.57it/s][A
    batches:  83%|████████▎ | 3099/3750 [00:04<00:00, 694.21it/s][A
    batches:  85%|████████▍ | 3169/3750 [00:04<00:00, 689.07it/s][A
    batches:  86%|████████▋ | 3239/3750 [00:04<00:00, 686.32it/s][A
    batches:  88%|████████▊ | 3308/3750 [00:04<00:00, 656.74it/s][A
    batches:  90%|█████████ | 3380/3750 [00:04<00:00, 673.45it/s][A
    batches:  92%|█████████▏| 3452/3750 [00:04<00:00, 685.07it/s][A
    batches:  94%|█████████▍| 3524/3750 [00:04<00:00, 693.00it/s][A
    batches:  96%|█████████▌| 3594/3750 [00:05<00:00, 675.03it/s][A
    batches:  98%|█████████▊| 3666/3750 [00:05<00:00, 687.06it/s][A
    batches: 100%|█████████▉| 3740/3750 [00:05<00:00, 683.91it/s][A
    epochs:  10%|█         | 3/30 [00:58<06:41, 14.87s/it]       [A
    batches:   0%|          | 0/3750 [00:00<?, ?it/s][A
    batches:   2%|▏         | 57/3750 [00:00<00:06, 569.66it/s][A
    batches:   3%|▎         | 131/3750 [00:00<00:05, 668.34it/s][A
    batches:   5%|▌         | 202/3750 [00:00<00:05, 687.28it/s][A
    batches:   7%|▋         | 275/3750 [00:00<00:05, 652.52it/s][A
    batches:   9%|▉         | 344/3750 [00:00<00:05, 665.35it/s][A
    batches:  11%|█         | 417/3750 [00:00<00:04, 684.92it/s][A
    batches:  13%|█▎        | 491/3750 [00:00<00:04, 699.77it/s][A
    batches:  15%|█▌        | 564/3750 [00:00<00:04, 707.99it/s][A
    batches:  17%|█▋        | 638/3750 [00:00<00:04, 714.96it/s][A
    batches:  19%|█▉        | 710/3750 [00:01<00:04, 714.52it/s][A
    batches:  21%|██        | 782/3750 [00:01<00:04, 713.74it/s][A
    batches:  23%|██▎       | 856/3750 [00:01<00:04, 720.71it/s][A
    batches:  25%|██▍       | 929/3750 [00:01<00:03, 719.96it/s][A
    batches:  27%|██▋       | 1002/3750 [00:01<00:03, 715.80it/s][A
    batches:  29%|██▊       | 1074/3750 [00:01<00:03, 713.11it/s][A
    batches:  31%|███       | 1147/3750 [00:01<00:03, 715.46it/s][A
    batches:  33%|███▎      | 1221/3750 [00:01<00:03, 720.57it/s][A
    batches:  35%|███▍      | 1294/3750 [00:01<00:03, 714.96it/s][A
    batches:  36%|███▋      | 1367/3750 [00:01<00:03, 717.04it/s][A
    batches:  38%|███▊      | 1439/3750 [00:02<00:03, 708.15it/s][A
    batches:  40%|████      | 1512/3750 [00:02<00:03, 713.94it/s][A
    batches:  42%|████▏     | 1587/3750 [00:02<00:02, 724.11it/s][A
    batches:  44%|████▍     | 1660/3750 [00:02<00:03, 696.22it/s][A
    batches:  46%|████▌     | 1732/3750 [00:02<00:02, 702.84it/s][A
    batches:  48%|████▊     | 1808/3750 [00:02<00:02, 717.93it/s][A
    batches:  50%|█████     | 1884/3750 [00:02<00:02, 727.84it/s][A
    batches:  52%|█████▏    | 1960/3750 [00:02<00:02, 734.82it/s][A
    batches:  54%|█████▍    | 2035/3750 [00:02<00:02, 738.65it/s][A
    batches:  56%|█████▋    | 2111/3750 [00:02<00:02, 742.73it/s][A
    batches:  58%|█████▊    | 2186/3750 [00:03<00:02, 744.80it/s][A
    batches:  60%|██████    | 2262/3750 [00:03<00:01, 746.74it/s][A
    batches:  62%|██████▏   | 2337/3750 [00:03<00:01, 745.75it/s][A
    batches:  64%|██████▍   | 2412/3750 [00:03<00:01, 710.69it/s][A
    batches:  66%|██████▋   | 2487/3750 [00:03<00:01, 719.86it/s][A
    batches:  68%|██████▊   | 2560/3750 [00:03<00:01, 695.47it/s][A
    batches:  70%|███████   | 2632/3750 [00:03<00:01, 701.82it/s][A
    batches:  72%|███████▏  | 2703/3750 [00:03<00:01, 702.82it/s][A
    batches:  74%|███████▍  | 2774/3750 [00:03<00:01, 685.21it/s][A
    batches:  76%|███████▌  | 2849/3750 [00:04<00:01, 701.34it/s][A
    batches:  78%|███████▊  | 2920/3750 [00:04<00:01, 701.51it/s][A
    batches:  80%|███████▉  | 2991/3750 [00:04<00:01, 703.07it/s][A
    batches:  82%|████████▏ | 3066/3750 [00:04<00:00, 715.65it/s][A
    batches:  84%|████████▍ | 3141/3750 [00:04<00:00, 725.25it/s][A
    batches:  86%|████████▌ | 3217/3750 [00:04<00:00, 733.85it/s][A
    batches:  88%|████████▊ | 3293/3750 [00:04<00:00, 738.99it/s][A
    batches:  90%|████████▉ | 3369/3750 [00:04<00:00, 742.41it/s][A
    batches:  92%|█████████▏| 3445/3750 [00:04<00:00, 747.33it/s][A
    batches:  94%|█████████▍| 3521/3750 [00:04<00:00, 750.13it/s][A
    batches:  96%|█████████▌| 3597/3750 [00:05<00:00, 751.52it/s][A
    batches:  98%|█████████▊| 3673/3750 [00:05<00:00, 751.97it/s][A
    epochs:  13%|█▎        | 4/30 [01:04<04:47, 11.06s/it]       [A
    batches:   0%|          | 0/3750 [00:00<?, ?it/s][A
    batches:   1%|▏         | 55/3750 [00:00<00:06, 546.64it/s][A
    batches:   3%|▎         | 123/3750 [00:00<00:05, 624.33it/s][A
    batches:   5%|▌         | 195/3750 [00:00<00:05, 665.98it/s][A
    batches:   7%|▋         | 262/3750 [00:00<00:05, 657.55it/s][A
    batches:   9%|▉         | 335/3750 [00:00<00:05, 681.57it/s][A
    batches:  11%|█         | 408/3750 [00:00<00:04, 692.68it/s][A
    batches:  13%|█▎        | 481/3750 [00:00<00:04, 701.91it/s][A
    batches:  15%|█▍        | 552/3750 [00:00<00:04, 666.52it/s][A
    batches:  17%|█▋        | 627/3750 [00:00<00:04, 690.75it/s][A
    batches:  19%|█▊        | 702/3750 [00:01<00:04, 706.49it/s][A
    batches:  21%|██        | 773/3750 [00:01<00:04, 707.10it/s][A
    batches:  23%|██▎       | 847/3750 [00:01<00:04, 715.12it/s][A
    batches:  25%|██▍       | 921/3750 [00:01<00:03, 720.77it/s][A
    batches:  27%|██▋       | 994/3750 [00:01<00:03, 715.31it/s][A
    batches:  28%|██▊       | 1066/3750 [00:01<00:03, 698.23it/s][A
    batches:  30%|███       | 1142/3750 [00:01<00:03, 713.59it/s][A
    batches:  32%|███▏      | 1217/3750 [00:01<00:03, 722.08it/s][A
    batches:  34%|███▍      | 1292/3750 [00:01<00:03, 728.03it/s][A
    batches:  36%|███▋      | 1368/3750 [00:01<00:03, 736.22it/s][A
    batches:  38%|███▊      | 1442/3750 [00:02<00:03, 736.92it/s][A
    batches:  40%|████      | 1516/3750 [00:02<00:03, 731.08it/s][A
    batches:  42%|████▏     | 1590/3750 [00:02<00:02, 733.51it/s][A
    batches:  44%|████▍     | 1665/3750 [00:02<00:02, 735.99it/s][A
    batches:  46%|████▋     | 1739/3750 [00:02<00:02, 729.33it/s][A
    batches:  48%|████▊     | 1812/3750 [00:02<00:02, 726.64it/s][A
    batches:  50%|█████     | 1885/3750 [00:02<00:02, 706.21it/s][A
    batches:  52%|█████▏    | 1959/3750 [00:02<00:02, 713.41it/s][A
    batches:  54%|█████▍    | 2031/3750 [00:02<00:02, 704.07it/s][A
    batches:  56%|█████▌    | 2104/3750 [00:02<00:02, 711.12it/s][A
    batches:  58%|█████▊    | 2178/3750 [00:03<00:02, 719.24it/s][A
    batches:  60%|██████    | 2252/3750 [00:03<00:02, 725.37it/s][A
    batches:  62%|██████▏   | 2325/3750 [00:03<00:02, 709.29it/s][A
    batches:  64%|██████▍   | 2399/3750 [00:03<00:01, 715.32it/s][A
    batches:  66%|██████▌   | 2474/3750 [00:03<00:01, 722.67it/s][A
    batches:  68%|██████▊   | 2548/3750 [00:03<00:01, 727.70it/s][A
    batches:  70%|██████▉   | 2621/3750 [00:03<00:02, 511.83it/s][A
    batches:  72%|███████▏  | 2682/3750 [00:03<00:02, 526.22it/s][A
    batches:  73%|███████▎  | 2742/3750 [00:04<00:02, 359.96it/s][A
    batches:  74%|███████▍  | 2793/3750 [00:04<00:02, 387.79it/s][A
    batches:  76%|███████▌  | 2851/3750 [00:04<00:02, 428.17it/s][A
    batches:  77%|███████▋  | 2905/3750 [00:04<00:01, 453.70it/s][A
    batches:  79%|███████▉  | 2958/3750 [00:04<00:01, 471.30it/s][A
    batches:  80%|████████  | 3014/3750 [00:04<00:01, 491.97it/s][A
    batches:  82%|████████▏ | 3067/3750 [00:04<00:01, 456.29it/s][A
    batches:  83%|████████▎ | 3119/3750 [00:04<00:01, 472.07it/s][A
    batches:  85%|████████▍ | 3169/3750 [00:05<00:01, 466.62it/s][A
    batches:  86%|████████▌ | 3221/3750 [00:05<00:01, 479.69it/s][A
    batches:  87%|████████▋ | 3279/3750 [00:05<00:00, 506.87it/s][A
    batches:  89%|████████▉ | 3343/3750 [00:05<00:00, 542.94it/s][A
    batches:  91%|█████████ | 3408/3750 [00:05<00:00, 571.91it/s][A
    batches:  93%|█████████▎| 3474/3750 [00:05<00:00, 597.54it/s][A
    batches:  94%|█████████▍| 3539/3750 [00:05<00:00, 606.28it/s][A
    batches:  96%|█████████▌| 3601/3750 [00:05<00:00, 606.82it/s][A
    batches:  98%|█████████▊| 3666/3750 [00:05<00:00, 617.57it/s][A
    batches:  99%|█████████▉| 3728/3750 [00:06<00:00, 617.27it/s][A
    epochs:  17%|█▋        | 5/30 [01:10<03:51,  9.25s/it]       [A
    batches:   0%|          | 0/3750 [00:00<?, ?it/s][A
    batches:   1%|▏         | 56/3750 [00:00<00:06, 556.72it/s][A
    batches:   3%|▎         | 123/3750 [00:00<00:05, 620.93it/s][A
    batches:   5%|▌         | 194/3750 [00:00<00:05, 658.81it/s][A
    batches:   7%|▋         | 260/3750 [00:00<00:05, 648.65it/s][A
    batches:   9%|▊         | 325/3750 [00:00<00:05, 642.05it/s][A
    batches:  11%|█         | 395/3750 [00:00<00:05, 659.31it/s][A
    batches:  12%|█▏        | 461/3750 [00:00<00:05, 652.20it/s][A
    batches:  14%|█▍        | 531/3750 [00:00<00:04, 665.30it/s][A
    batches:  16%|█▌        | 600/3750 [00:00<00:04, 670.94it/s][A
    batches:  18%|█▊        | 669/3750 [00:01<00:04, 676.51it/s][A
    batches:  20%|█▉        | 737/3750 [00:01<00:04, 669.53it/s][A
    batches:  21%|██▏       | 804/3750 [00:01<00:04, 654.47it/s][A
    batches:  23%|██▎       | 875/3750 [00:01<00:04, 668.76it/s][A
    batches:  25%|██▌       | 943/3750 [00:01<00:04, 670.73it/s][A
    batches:  27%|██▋       | 1012/3750 [00:01<00:04, 673.82it/s][A
    batches:  29%|██▉       | 1085/3750 [00:01<00:03, 689.67it/s][A
    batches:  31%|███       | 1155/3750 [00:01<00:03, 690.89it/s][A
    batches:  33%|███▎      | 1225/3750 [00:01<00:03, 669.41it/s][A
    batches:  35%|███▍      | 1299/3750 [00:01<00:03, 689.75it/s][A
    batches:  37%|███▋      | 1373/3750 [00:02<00:03, 703.22it/s][A
    batches:  39%|███▊      | 1447/3750 [00:02<00:03, 713.65it/s][A
    batches:  41%|████      | 1519/3750 [00:02<00:03, 705.16it/s][A
    batches:  42%|████▏     | 1590/3750 [00:02<00:03, 705.83it/s][A
    batches:  44%|████▍     | 1665/3750 [00:02<00:02, 717.77it/s][A
    batches:  46%|████▋     | 1737/3750 [00:02<00:02, 715.38it/s][A
    batches:  48%|████▊     | 1809/3750 [00:02<00:02, 680.78it/s][A
    batches:  50%|█████     | 1882/3750 [00:02<00:02, 693.99it/s][A
    batches:  52%|█████▏    | 1956/3750 [00:02<00:02, 706.43it/s][A
    batches:  54%|█████▍    | 2027/3750 [00:02<00:02, 679.42it/s][A
    batches:  56%|█████▌    | 2096/3750 [00:03<00:02, 667.12it/s][A
    batches:  58%|█████▊    | 2166/3750 [00:03<00:02, 675.98it/s][A
    batches:  60%|█████▉    | 2237/3750 [00:03<00:02, 684.59it/s][A
    batches:  62%|██████▏   | 2308/3750 [00:03<00:02, 691.10it/s][A
    batches:  63%|██████▎   | 2380/3750 [00:03<00:01, 697.88it/s][A
    batches:  65%|██████▌   | 2452/3750 [00:03<00:01, 701.83it/s][A
    batches:  67%|██████▋   | 2524/3750 [00:03<00:01, 705.18it/s][A
    batches:  69%|██████▉   | 2595/3750 [00:03<00:01, 706.31it/s][A
    batches:  71%|███████   | 2667/3750 [00:03<00:01, 708.70it/s][A
    batches:  73%|███████▎  | 2738/3750 [00:03<00:01, 707.81it/s][A
    batches:  75%|███████▍  | 2809/3750 [00:04<00:01, 700.84it/s][A
    batches:  77%|███████▋  | 2880/3750 [00:04<00:01, 703.17it/s][A
    batches:  79%|███████▊  | 2952/3750 [00:04<00:01, 705.87it/s][A
    batches:  81%|████████  | 3023/3750 [00:04<00:01, 703.91it/s][A
    batches:  83%|████████▎ | 3094/3750 [00:04<00:00, 703.92it/s][A
    batches:  84%|████████▍ | 3166/3750 [00:04<00:00, 706.80it/s][A
    batches:  86%|████████▋ | 3238/3750 [00:04<00:00, 708.35it/s][A
    batches:  88%|████████▊ | 3309/3750 [00:04<00:00, 707.35it/s][A
    batches:  90%|█████████ | 3381/3750 [00:04<00:00, 709.76it/s][A
    batches:  92%|█████████▏| 3452/3750 [00:05<00:00, 708.35it/s][A
    batches:  94%|█████████▍| 3525/3750 [00:05<00:00, 712.62it/s][A
    batches:  96%|█████████▌| 3597/3750 [00:05<00:00, 710.60it/s][A
    batches:  98%|█████████▊| 3669/3750 [00:05<00:00, 710.08it/s][A
    batches: 100%|█████████▉| 3745/3750 [00:05<00:00, 722.83it/s][A
    epochs:  20%|██        | 6/30 [01:15<03:10,  7.95s/it]       [A
    batches:   0%|          | 0/3750 [00:00<?, ?it/s][A
    batches:   2%|▏         | 60/3750 [00:00<00:06, 591.62it/s][A
    batches:   4%|▎         | 138/3750 [00:00<00:05, 698.31it/s][A
    batches:   6%|▌         | 215/3750 [00:00<00:04, 728.99it/s][A
    batches:   8%|▊         | 292/3750 [00:00<00:04, 742.67it/s][A
    batches:  10%|▉         | 369/3750 [00:00<00:04, 750.85it/s][A
    batches:  12%|█▏        | 446/3750 [00:00<00:04, 754.81it/s][A
    batches:  14%|█▍        | 522/3750 [00:00<00:04, 756.06it/s][A
    batches:  16%|█▌        | 599/3750 [00:00<00:04, 759.49it/s][A
    batches:  18%|█▊        | 677/3750 [00:00<00:04, 762.72it/s][A
    batches:  20%|██        | 755/3750 [00:01<00:03, 764.91it/s][A
    batches:  22%|██▏       | 832/3750 [00:01<00:03, 764.82it/s][A
    batches:  24%|██▍       | 909/3750 [00:01<00:03, 764.89it/s][A
    batches:  26%|██▋       | 986/3750 [00:01<00:03, 764.43it/s][A
    batches:  28%|██▊       | 1063/3750 [00:01<00:03, 765.23it/s][A
    batches:  30%|███       | 1141/3750 [00:01<00:03, 766.97it/s][A
    batches:  32%|███▏      | 1218/3750 [00:01<00:03, 767.47it/s][A
    batches:  35%|███▍      | 1295/3750 [00:01<00:03, 765.27it/s][A
    batches:  37%|███▋      | 1373/3750 [00:01<00:03, 769.59it/s][A
    batches:  39%|███▊      | 1450/3750 [00:01<00:03, 741.18it/s][A
    batches:  41%|████      | 1525/3750 [00:02<00:03, 717.80it/s][A
    batches:  43%|████▎     | 1598/3750 [00:02<00:03, 707.59it/s][A
    batches:  45%|████▍     | 1669/3750 [00:02<00:02, 703.61it/s][A
    batches:  46%|████▋     | 1740/3750 [00:02<00:02, 700.80it/s][A
    batches:  48%|████▊     | 1814/3750 [00:02<00:02, 710.51it/s][A
    batches:  50%|█████     | 1890/3750 [00:02<00:02, 721.80it/s][A
    batches:  52%|█████▏    | 1967/3750 [00:02<00:02, 733.73it/s][A
    batches:  55%|█████▍    | 2044/3750 [00:02<00:02, 742.24it/s][A
    batches:  57%|█████▋    | 2120/3750 [00:02<00:02, 747.19it/s][A
    batches:  59%|█████▊    | 2196/3750 [00:02<00:02, 750.10it/s][A
    batches:  61%|██████    | 2272/3750 [00:03<00:01, 741.82it/s][A
    batches:  63%|██████▎   | 2347/3750 [00:03<00:01, 723.41it/s][A
    batches:  65%|██████▍   | 2420/3750 [00:03<00:01, 712.30it/s][A
    batches:  67%|██████▋   | 2498/3750 [00:03<00:01, 729.90it/s][A
    batches:  69%|██████▊   | 2576/3750 [00:03<00:01, 742.22it/s][A
    batches:  71%|███████   | 2654/3750 [00:03<00:01, 750.45it/s][A
    batches:  73%|███████▎  | 2730/3750 [00:03<00:01, 752.74it/s][A
    batches:  75%|███████▍  | 2809/3750 [00:03<00:01, 761.74it/s][A
    batches:  77%|███████▋  | 2886/3750 [00:03<00:01, 762.81it/s][A
    batches:  79%|███████▉  | 2964/3750 [00:03<00:01, 765.62it/s][A
    batches:  81%|████████  | 3041/3750 [00:04<00:00, 761.14it/s][A
    batches:  83%|████████▎ | 3121/3750 [00:04<00:00, 770.60it/s][A
    batches:  85%|████████▌ | 3199/3750 [00:04<00:00, 769.96it/s][A
    batches:  87%|████████▋ | 3277/3750 [00:04<00:00, 770.90it/s][A
    batches:  89%|████████▉ | 3355/3750 [00:04<00:00, 771.77it/s][A
    batches:  92%|█████████▏| 3433/3750 [00:04<00:00, 770.10it/s][A
    batches:  94%|█████████▎| 3511/3750 [00:04<00:00, 766.03it/s][A
    batches:  96%|█████████▌| 3589/3750 [00:04<00:00, 767.44it/s][A
    batches:  98%|█████████▊| 3666/3750 [00:04<00:00, 767.02it/s][A
    batches: 100%|█████████▉| 3747/3750 [00:04<00:00, 777.39it/s][A
    epochs:  23%|██▎       | 7/30 [01:20<02:40,  6.99s/it]       [A
    batches:   0%|          | 0/3750 [00:00<?, ?it/s][A
    batches:   2%|▏         | 61/3750 [00:00<00:06, 604.74it/s][A
    batches:   4%|▎         | 133/3750 [00:00<00:05, 671.66it/s][A
    batches:   5%|▌         | 206/3750 [00:00<00:05, 695.08it/s][A
    batches:   7%|▋         | 279/3750 [00:00<00:04, 707.19it/s][A
    batches:   9%|▉         | 353/3750 [00:00<00:04, 716.05it/s][A
    batches:  11%|█▏        | 426/3750 [00:00<00:04, 720.49it/s][A
    batches:  13%|█▎        | 499/3750 [00:00<00:04, 720.66it/s][A
    batches:  15%|█▌        | 572/3750 [00:00<00:04, 696.10it/s][A
    batches:  17%|█▋        | 643/3750 [00:00<00:04, 698.13it/s][A
    batches:  19%|█▉        | 717/3750 [00:01<00:04, 708.32it/s][A
    batches:  21%|██        | 790/3750 [00:01<00:04, 712.10it/s][A
    batches:  23%|██▎       | 863/3750 [00:01<00:04, 714.41it/s][A
    batches:  25%|██▍       | 935/3750 [00:01<00:04, 702.38it/s][A
    batches:  27%|██▋       | 1007/3750 [00:01<00:03, 707.41it/s][A
    batches:  29%|██▉       | 1080/3750 [00:01<00:03, 711.69it/s][A
    batches:  31%|███       | 1152/3750 [00:01<00:03, 683.77it/s][A
    batches:  33%|███▎      | 1221/3750 [00:01<00:03, 669.15it/s][A
    batches:  35%|███▍      | 1294/3750 [00:01<00:03, 685.99it/s][A
    batches:  36%|███▋      | 1363/3750 [00:01<00:03, 672.87it/s][A
    batches:  38%|███▊      | 1436/3750 [00:02<00:03, 686.99it/s][A
    batches:  40%|████      | 1506/3750 [00:02<00:03, 690.59it/s][A
    batches:  42%|████▏     | 1576/3750 [00:02<00:03, 656.59it/s][A
    batches:  44%|████▍     | 1644/3750 [00:02<00:03, 662.03it/s][A
    batches:  46%|████▌     | 1712/3750 [00:02<00:03, 666.92it/s][A
    batches:  47%|████▋     | 1780/3750 [00:02<00:02, 669.50it/s][A
    batches:  49%|████▉     | 1848/3750 [00:02<00:02, 672.07it/s][A
    batches:  51%|█████     | 1917/3750 [00:02<00:02, 675.68it/s][A
    batches:  53%|█████▎    | 1986/3750 [00:02<00:02, 678.11it/s][A
    batches:  55%|█████▍    | 2056/3750 [00:02<00:02, 683.84it/s][A
    batches:  57%|█████▋    | 2126/3750 [00:03<00:02, 688.14it/s][A
    batches:  59%|█████▊    | 2196/3750 [00:03<00:02, 689.36it/s][A
    batches:  60%|██████    | 2266/3750 [00:03<00:02, 691.03it/s][A
    batches:  62%|██████▏   | 2336/3750 [00:03<00:02, 692.68it/s][A
    batches:  64%|██████▍   | 2406/3750 [00:03<00:01, 690.11it/s][A
    batches:  66%|██████▌   | 2478/3750 [00:03<00:01, 698.94it/s][A
    batches:  68%|██████▊   | 2548/3750 [00:03<00:01, 699.01it/s][A
    batches:  70%|██████▉   | 2620/3750 [00:03<00:01, 702.51it/s][A
    batches:  72%|███████▏  | 2692/3750 [00:03<00:01, 704.89it/s][A
    batches:  74%|███████▎  | 2764/3750 [00:03<00:01, 708.84it/s][A
    batches:  76%|███████▌  | 2837/3750 [00:04<00:01, 713.58it/s][A
    batches:  78%|███████▊  | 2910/3750 [00:04<00:01, 717.44it/s][A
    batches:  80%|███████▉  | 2982/3750 [00:04<00:01, 717.28it/s][A
    batches:  81%|████████▏ | 3056/3750 [00:04<00:00, 721.10it/s][A
    batches:  83%|████████▎ | 3129/3750 [00:04<00:00, 723.18it/s][A
    batches:  85%|████████▌ | 3202/3750 [00:04<00:00, 722.43it/s][A
    batches:  87%|████████▋ | 3275/3750 [00:04<00:00, 722.28it/s][A
    batches:  89%|████████▉ | 3348/3750 [00:04<00:00, 721.27it/s][A
    batches:  91%|█████████ | 3421/3750 [00:04<00:00, 720.15it/s][A
    batches:  93%|█████████▎| 3494/3750 [00:05<00:00, 721.36it/s][A
    batches:  95%|█████████▌| 3567/3750 [00:05<00:00, 719.11it/s][A
    batches:  97%|█████████▋| 3645/3750 [00:05<00:00, 734.84it/s][A
    batches:  99%|█████████▉| 3723/3750 [00:05<00:00, 747.93it/s][A
    epochs:  27%|██▋       | 8/30 [01:25<02:22,  6.46s/it]       [A
    batches:   0%|          | 0/3750 [00:00<?, ?it/s][A
    batches:   2%|▏         | 57/3750 [00:00<00:06, 556.86it/s][A
    batches:   4%|▎         | 132/3750 [00:00<00:05, 668.55it/s][A
    batches:   5%|▌         | 202/3750 [00:00<00:05, 681.37it/s][A
    batches:   7%|▋         | 275/3750 [00:00<00:04, 699.75it/s][A
    batches:   9%|▉         | 349/3750 [00:00<00:04, 711.50it/s][A
    batches:  11%|█▏        | 422/3750 [00:00<00:04, 717.39it/s][A
    batches:  13%|█▎        | 495/3750 [00:00<00:04, 714.74it/s][A
    batches:  15%|█▌        | 567/3750 [00:00<00:04, 712.66it/s][A
    batches:  17%|█▋        | 639/3750 [00:00<00:04, 704.49it/s][A
    batches:  19%|█▉        | 711/3750 [00:01<00:04, 709.05it/s][A
    batches:  21%|██        | 784/3750 [00:01<00:04, 713.37it/s][A
    batches:  23%|██▎       | 856/3750 [00:01<00:04, 714.85it/s][A
    batches:  25%|██▍       | 929/3750 [00:01<00:03, 716.87it/s][A
    batches:  27%|██▋       | 1006/3750 [00:01<00:03, 732.41it/s][A
    batches:  29%|██▉       | 1083/3750 [00:01<00:03, 742.56it/s][A
    batches:  31%|███       | 1158/3750 [00:01<00:03, 705.65it/s][A
    batches:  33%|███▎      | 1229/3750 [00:01<00:03, 685.39it/s][A
    batches:  35%|███▍      | 1298/3750 [00:01<00:03, 682.72it/s][A
    batches:  36%|███▋      | 1367/3750 [00:01<00:03, 678.76it/s][A
    batches:  38%|███▊      | 1437/3750 [00:02<00:03, 682.42it/s][A
    batches:  40%|████      | 1509/3750 [00:02<00:03, 693.12it/s][A
    batches:  42%|████▏     | 1581/3750 [00:02<00:03, 700.23it/s][A
    batches:  44%|████▍     | 1653/3750 [00:02<00:02, 704.17it/s][A
    batches:  46%|████▌     | 1726/3750 [00:02<00:02, 709.53it/s][A
    batches:  48%|████▊     | 1800/3750 [00:02<00:02, 716.71it/s][A
    batches:  50%|████▉     | 1873/3750 [00:02<00:02, 718.04it/s][A
    batches:  52%|█████▏    | 1945/3750 [00:02<00:02, 709.33it/s][A
    batches:  54%|█████▍    | 2018/3750 [00:02<00:02, 713.72it/s][A
    batches:  56%|█████▌    | 2092/3750 [00:02<00:02, 719.42it/s][A
    batches:  58%|█████▊    | 2164/3750 [00:03<00:02, 711.07it/s][A
    batches:  60%|█████▉    | 2238/3750 [00:03<00:02, 718.58it/s][A
    batches:  62%|██████▏   | 2312/3750 [00:03<00:01, 723.59it/s][A
    batches:  64%|██████▎   | 2386/3750 [00:03<00:01, 727.75it/s][A
    batches:  66%|██████▌   | 2460/3750 [00:03<00:01, 728.81it/s][A
    batches:  68%|██████▊   | 2534/3750 [00:03<00:01, 729.75it/s][A
    batches:  70%|██████▉   | 2608/3750 [00:03<00:01, 731.38it/s][A
    batches:  72%|███████▏  | 2682/3750 [00:03<00:01, 731.17it/s][A
    batches:  73%|███████▎  | 2756/3750 [00:03<00:01, 731.70it/s][A
    batches:  75%|███████▌  | 2830/3750 [00:03<00:01, 732.83it/s][A
    batches:  77%|███████▋  | 2905/3750 [00:04<00:01, 735.62it/s][A
    batches:  79%|███████▉  | 2979/3750 [00:04<00:01, 733.79it/s][A
    batches:  81%|████████▏ | 3053/3750 [00:04<00:00, 734.14it/s][A
    batches:  83%|████████▎ | 3127/3750 [00:04<00:00, 722.69it/s][A
    batches:  85%|████████▌ | 3201/3750 [00:04<00:00, 726.27it/s][A
    batches:  87%|████████▋ | 3274/3750 [00:04<00:00, 727.01it/s][A
    batches:  89%|████████▉ | 3347/3750 [00:04<00:00, 704.89it/s][A
    batches:  91%|█████████▏| 3422/3750 [00:04<00:00, 716.53it/s][A
    batches:  93%|█████████▎| 3494/3750 [00:04<00:00, 713.88it/s][A
    batches:  95%|█████████▌| 3568/3750 [00:04<00:00, 719.91it/s][A
    batches:  97%|█████████▋| 3643/3750 [00:05<00:00, 728.37it/s][A
    batches:  99%|█████████▉| 3716/3750 [00:05<00:00, 727.26it/s][A
    epochs:  30%|███       | 9/30 [01:31<02:07,  6.08s/it]       [A
    batches:   0%|          | 0/3750 [00:00<?, ?it/s][A
    batches:   2%|▏         | 58/3750 [00:00<00:06, 573.14it/s][A
    batches:   3%|▎         | 128/3750 [00:00<00:05, 645.70it/s][A
    batches:   5%|▌         | 193/3750 [00:00<00:05, 639.22it/s][A
    batches:   7%|▋         | 262/3750 [00:00<00:05, 658.84it/s][A
    batches:   9%|▊         | 328/3750 [00:00<00:05, 656.97it/s][A
    batches:  11%|█         | 394/3750 [00:00<00:05, 657.83it/s][A
    batches:  12%|█▏        | 464/3750 [00:00<00:04, 671.23it/s][A
    batches:  14%|█▍        | 540/3750 [00:00<00:04, 696.76it/s][A
    batches:  16%|█▋        | 611/3750 [00:00<00:04, 698.18it/s][A
    batches:  18%|█▊        | 682/3750 [00:01<00:04, 699.13it/s][A
    batches:  20%|██        | 756/3750 [00:01<00:04, 709.86it/s][A
    batches:  22%|██▏       | 827/3750 [00:01<00:04, 680.44it/s][A
    batches:  24%|██▍       | 896/3750 [00:01<00:04, 666.38it/s][A
    batches:  26%|██▌       | 969/3750 [00:01<00:04, 683.48it/s][A
    batches:  28%|██▊       | 1042/3750 [00:01<00:03, 694.95it/s][A
    batches:  30%|██▉       | 1112/3750 [00:01<00:03, 666.34it/s][A
    batches:  31%|███▏      | 1179/3750 [00:01<00:03, 643.14it/s][A
    batches:  33%|███▎      | 1244/3750 [00:01<00:03, 630.56it/s][A
    batches:  35%|███▌      | 1317/3750 [00:01<00:03, 656.55it/s][A
    batches:  37%|███▋      | 1389/3750 [00:02<00:03, 674.41it/s][A
    batches:  39%|███▉      | 1466/3750 [00:02<00:03, 701.40it/s][A
    batches:  41%|████      | 1543/3750 [00:02<00:03, 721.50it/s][A
    batches:  43%|████▎     | 1616/3750 [00:02<00:03, 710.88it/s][A
    batches:  45%|████▌     | 1690/3750 [00:02<00:02, 717.29it/s][A
    batches:  47%|████▋     | 1764/3750 [00:02<00:02, 721.66it/s][A
    batches:  49%|████▉     | 1837/3750 [00:02<00:02, 721.37it/s][A
    batches:  51%|█████     | 1910/3750 [00:02<00:02, 723.29it/s][A
    batches:  53%|█████▎    | 1983/3750 [00:02<00:02, 722.76it/s][A
    batches:  55%|█████▍    | 2056/3750 [00:02<00:02, 711.16it/s][A
    batches:  57%|█████▋    | 2128/3750 [00:03<00:02, 702.10it/s][A
    batches:  59%|█████▊    | 2202/3750 [00:03<00:02, 712.38it/s][A
    batches:  61%|██████    | 2276/3750 [00:03<00:02, 718.69it/s][A
    batches:  63%|██████▎   | 2348/3750 [00:03<00:01, 714.01it/s][A
    batches:  65%|██████▍   | 2422/3750 [00:03<00:01, 719.19it/s][A
    batches:  67%|██████▋   | 2496/3750 [00:03<00:01, 723.13it/s][A
    batches:  69%|██████▊   | 2570/3750 [00:03<00:01, 726.28it/s][A
    batches:  71%|███████   | 2644/3750 [00:03<00:01, 728.47it/s][A
    batches:  72%|███████▏  | 2718/3750 [00:03<00:01, 729.97it/s][A
    batches:  74%|███████▍  | 2792/3750 [00:04<00:01, 731.51it/s][A
    batches:  76%|███████▋  | 2866/3750 [00:04<00:01, 732.98it/s][A
    batches:  78%|███████▊  | 2940/3750 [00:04<00:01, 734.11it/s][A
    batches:  80%|████████  | 3014/3750 [00:04<00:01, 710.51it/s][A
    batches:  82%|████████▏ | 3086/3750 [00:04<00:00, 708.02it/s][A
    batches:  84%|████████▍ | 3160/3750 [00:04<00:00, 716.62it/s][A
    batches:  86%|████████▌ | 3232/3750 [00:04<00:00, 717.41it/s][A
    batches:  88%|████████▊ | 3308/3750 [00:04<00:00, 728.39it/s][A
    batches:  90%|█████████ | 3383/3750 [00:04<00:00, 732.11it/s][A
    batches:  92%|█████████▏| 3458/3750 [00:04<00:00, 734.46it/s][A
    batches:  94%|█████████▍| 3532/3750 [00:05<00:00, 734.75it/s][A
    batches:  96%|█████████▌| 3606/3750 [00:05<00:00, 732.94it/s][A
    batches:  98%|█████████▊| 3680/3750 [00:05<00:00, 733.79it/s][A
    epochs:  33%|███▎      | 10/30 [01:36<01:56,  5.85s/it]      [A
    batches:   0%|          | 0/3750 [00:00<?, ?it/s][A
    batches:   1%|▏         | 54/3750 [00:00<00:06, 538.84it/s][A
    batches:   3%|▎         | 118/3750 [00:00<00:06, 593.89it/s][A
    batches:   5%|▌         | 191/3750 [00:00<00:05, 651.67it/s][A
    batches:   7%|▋         | 264/3750 [00:00<00:05, 681.86it/s][A
    batches:   9%|▉         | 333/3750 [00:00<00:05, 671.11it/s][A
    batches:  11%|█         | 401/3750 [00:00<00:05, 668.38it/s][A
    batches:  13%|█▎        | 473/3750 [00:00<00:04, 682.57it/s][A
    batches:  15%|█▍        | 545/3750 [00:00<00:04, 692.91it/s][A
    batches:  16%|█▋        | 618/3750 [00:00<00:04, 701.96it/s][A
    batches:  18%|█▊        | 689/3750 [00:01<00:04, 696.07it/s][A
    batches:  20%|██        | 759/3750 [00:01<00:04, 656.74it/s][A
    batches:  22%|██▏       | 832/3750 [00:01<00:04, 676.10it/s][A
    batches:  24%|██▍       | 901/3750 [00:01<00:04, 638.20it/s][A
    batches:  26%|██▌       | 966/3750 [00:01<00:04, 639.56it/s][A
    batches:  28%|██▊       | 1032/3750 [00:01<00:04, 643.80it/s][A
    batches:  29%|██▉       | 1101/3750 [00:01<00:04, 655.38it/s][A
    batches:  31%|███       | 1168/3750 [00:01<00:03, 657.06it/s][A
    batches:  33%|███▎      | 1239/3750 [00:01<00:03, 671.03it/s][A
    batches:  35%|███▍      | 1311/3750 [00:01<00:03, 684.37it/s][A
    batches:  37%|███▋      | 1383/3750 [00:02<00:03, 692.83it/s][A
    batches:  39%|███▉      | 1456/3750 [00:02<00:03, 703.42it/s][A
    batches:  41%|████      | 1527/3750 [00:02<00:03, 693.51it/s][A
    batches:  43%|████▎     | 1601/3750 [00:02<00:03, 706.30it/s][A
    batches:  45%|████▍     | 1672/3750 [00:02<00:02, 703.27it/s][A
    batches:  47%|████▋     | 1744/3750 [00:02<00:02, 705.89it/s][A
    batches:  48%|████▊     | 1815/3750 [00:02<00:02, 702.43it/s][A
    batches:  50%|█████     | 1887/3750 [00:02<00:02, 706.76it/s][A
    batches:  52%|█████▏    | 1961/3750 [00:02<00:02, 714.37it/s][A
    batches:  54%|█████▍    | 2033/3750 [00:02<00:02, 715.65it/s][A
    batches:  56%|█████▌    | 2109/3750 [00:03<00:02, 727.25it/s][A
    batches:  58%|█████▊    | 2185/3750 [00:03<00:02, 736.68it/s][A
    batches:  60%|██████    | 2259/3750 [00:03<00:02, 723.37it/s][A
    batches:  62%|██████▏   | 2332/3750 [00:03<00:01, 721.87it/s][A
    batches:  64%|██████▍   | 2405/3750 [00:03<00:01, 723.31it/s][A
    batches:  66%|██████▌   | 2479/3750 [00:03<00:01, 725.40it/s][A
    batches:  68%|██████▊   | 2553/3750 [00:03<00:01, 728.24it/s][A
    batches:  70%|███████   | 2626/3750 [00:03<00:01, 727.58it/s][A
    batches:  72%|███████▏  | 2699/3750 [00:03<00:01, 728.14it/s][A
    batches:  74%|███████▍  | 2772/3750 [00:03<00:01, 719.75it/s][A
    batches:  76%|███████▌  | 2845/3750 [00:04<00:01, 721.34it/s][A
    batches:  78%|███████▊  | 2918/3750 [00:04<00:01, 719.76it/s][A
    batches:  80%|███████▉  | 2992/3750 [00:04<00:01, 725.22it/s][A
    batches:  82%|████████▏ | 3065/3750 [00:04<00:00, 725.89it/s][A
    batches:  84%|████████▎ | 3138/3750 [00:04<00:00, 720.01it/s][A
    batches:  86%|████████▌ | 3211/3750 [00:04<00:00, 701.92it/s][A
    batches:  88%|████████▊ | 3285/3750 [00:04<00:00, 711.92it/s][A
    batches:  90%|████████▉ | 3359/3750 [00:04<00:00, 717.48it/s][A
    batches:  92%|█████████▏| 3433/3750 [00:04<00:00, 722.33it/s][A
    batches:  94%|█████████▎| 3507/3750 [00:05<00:00, 724.43it/s][A
    batches:  95%|█████████▌| 3580/3750 [00:05<00:00, 725.99it/s][A
    batches:  97%|█████████▋| 3653/3750 [00:05<00:00, 727.15it/s][A
    batches:  99%|█████████▉| 3726/3750 [00:05<00:00, 716.16it/s][A
    epochs:  37%|███▋      | 11/30 [01:41<01:48,  5.70s/it]      [A
    batches:   0%|          | 0/3750 [00:00<?, ?it/s][A
    batches:   1%|▏         | 56/3750 [00:00<00:06, 559.17it/s][A
    batches:   4%|▎         | 136/3750 [00:00<00:05, 697.92it/s][A
    batches:   6%|▌         | 214/3750 [00:00<00:04, 732.77it/s][A
    batches:   8%|▊         | 291/3750 [00:00<00:04, 744.42it/s][A
    batches:  10%|▉         | 368/3750 [00:00<00:04, 751.98it/s][A
    batches:  12%|█▏        | 446/3750 [00:00<00:04, 761.40it/s][A
    batches:  14%|█▍        | 524/3750 [00:00<00:04, 765.98it/s][A
    batches:  16%|█▌        | 602/3750 [00:00<00:04, 768.76it/s][A
    batches:  18%|█▊        | 681/3750 [00:00<00:03, 773.29it/s][A
    batches:  20%|██        | 762/3750 [00:01<00:03, 780.07it/s][A
    batches:  23%|██▎       | 845/3750 [00:01<00:03, 792.12it/s][A
    batches:  25%|██▍       | 927/3750 [00:01<00:03, 800.34it/s][A
    batches:  27%|██▋       | 1008/3750 [00:01<00:03, 791.59it/s][A
    batches:  29%|██▉       | 1088/3750 [00:01<00:03, 793.69it/s][A
    batches:  31%|███       | 1168/3750 [00:01<00:03, 783.41it/s][A
    batches:  33%|███▎      | 1247/3750 [00:01<00:03, 784.57it/s][A
    batches:  35%|███▌      | 1326/3750 [00:01<00:03, 783.68it/s][A
    batches:  37%|███▋      | 1405/3750 [00:01<00:02, 785.39it/s][A
    batches:  40%|███▉      | 1484/3750 [00:01<00:02, 785.79it/s][A
    batches:  42%|████▏     | 1563/3750 [00:02<00:02, 786.07it/s][A
    batches:  44%|████▍     | 1643/3750 [00:02<00:02, 787.96it/s][A
    batches:  46%|████▌     | 1722/3750 [00:02<00:02, 788.29it/s][A
    batches:  48%|████▊     | 1801/3750 [00:02<00:02, 786.56it/s][A
    batches:  50%|█████     | 1880/3750 [00:02<00:02, 781.79it/s][A
    batches:  52%|█████▏    | 1959/3750 [00:02<00:02, 782.12it/s][A
    batches:  54%|█████▍    | 2038/3750 [00:02<00:02, 783.26it/s][A
    batches:  56%|█████▋    | 2117/3750 [00:02<00:02, 783.71it/s][A
    batches:  59%|█████▊    | 2197/3750 [00:02<00:01, 786.63it/s][A
    batches:  61%|██████    | 2276/3750 [00:02<00:01, 787.37it/s][A
    batches:  63%|██████▎   | 2355/3750 [00:03<00:01, 786.69it/s][A
    batches:  65%|██████▍   | 2435/3750 [00:03<00:01, 787.84it/s][A
    batches:  67%|██████▋   | 2514/3750 [00:03<00:01, 786.39it/s][A
    batches:  69%|██████▉   | 2594/3750 [00:03<00:01, 787.73it/s][A
    batches:  71%|███████▏  | 2673/3750 [00:03<00:01, 786.84it/s][A
    batches:  73%|███████▎  | 2752/3750 [00:03<00:01, 786.86it/s][A
    batches:  75%|███████▌  | 2831/3750 [00:03<00:01, 786.96it/s][A
    batches:  78%|███████▊  | 2910/3750 [00:03<00:01, 784.08it/s][A
    batches:  80%|███████▉  | 2989/3750 [00:03<00:00, 785.13it/s][A
    batches:  82%|████████▏ | 3068/3750 [00:03<00:00, 785.77it/s][A
    batches:  84%|████████▍ | 3148/3750 [00:04<00:00, 788.12it/s][A
    batches:  86%|████████▌ | 3227/3750 [00:04<00:00, 787.68it/s][A
    batches:  88%|████████▊ | 3306/3750 [00:04<00:00, 787.99it/s][A
    batches:  90%|█████████ | 3386/3750 [00:04<00:00, 788.97it/s][A
    batches:  92%|█████████▏| 3465/3750 [00:04<00:00, 782.88it/s][A
    batches:  95%|█████████▍| 3544/3750 [00:04<00:00, 782.12it/s][A
    batches:  97%|█████████▋| 3623/3750 [00:04<00:00, 784.29it/s][A
    batches:  99%|█████████▊| 3702/3750 [00:04<00:00, 783.45it/s][A
    epochs:  40%|████      | 12/30 [01:46<01:37,  5.43s/it]      [A
    batches:   0%|          | 0/3750 [00:00<?, ?it/s][A
    batches:   2%|▏         | 63/3750 [00:00<00:05, 625.47it/s][A
    batches:   4%|▎         | 137/3750 [00:00<00:05, 691.89it/s][A
    batches:   6%|▌         | 212/3750 [00:00<00:04, 716.36it/s][A
    batches:   8%|▊         | 287/3750 [00:00<00:04, 727.85it/s][A
    batches:  10%|▉         | 362/3750 [00:00<00:04, 735.13it/s][A
    batches:  12%|█▏        | 437/3750 [00:00<00:04, 739.30it/s][A
    batches:  14%|█▎        | 512/3750 [00:00<00:04, 739.83it/s][A
    batches:  16%|█▌        | 586/3750 [00:00<00:04, 717.65it/s][A
    batches:  18%|█▊        | 659/3750 [00:00<00:04, 721.32it/s][A
    batches:  20%|█▉        | 734/3750 [00:01<00:04, 728.82it/s][A
    batches:  22%|██▏       | 809/3750 [00:01<00:04, 733.46it/s][A
    batches:  24%|██▎       | 883/3750 [00:01<00:04, 714.80it/s][A
    batches:  25%|██▌       | 956/3750 [00:01<00:03, 717.77it/s][A
    batches:  27%|██▋       | 1031/3750 [00:01<00:03, 725.14it/s][A
    batches:  29%|██▉       | 1104/3750 [00:01<00:03, 720.77it/s][A
    batches:  31%|███▏      | 1177/3750 [00:01<00:03, 693.01it/s][A
    batches:  33%|███▎      | 1252/3750 [00:01<00:03, 709.35it/s][A
    batches:  35%|███▌      | 1327/3750 [00:01<00:03, 719.40it/s][A
    batches:  37%|███▋      | 1401/3750 [00:01<00:03, 724.65it/s][A
    batches:  39%|███▉      | 1474/3750 [00:02<00:03, 701.33it/s][A
    batches:  41%|████▏     | 1549/3750 [00:02<00:03, 713.82it/s][A
    batches:  43%|████▎     | 1622/3750 [00:02<00:02, 716.68it/s][A
    batches:  45%|████▌     | 1697/3750 [00:02<00:02, 724.82it/s][A
    batches:  47%|████▋     | 1770/3750 [00:02<00:02, 684.87it/s][A
    batches:  49%|████▉     | 1844/3750 [00:02<00:02, 699.16it/s][A
    batches:  51%|█████     | 1919/3750 [00:02<00:02, 711.77it/s][A
    batches:  53%|█████▎    | 1993/3750 [00:02<00:02, 719.01it/s][A
    batches:  55%|█████▌    | 2068/3750 [00:02<00:02, 727.75it/s][A
    batches:  57%|█████▋    | 2143/3750 [00:02<00:02, 732.99it/s][A
    batches:  59%|█████▉    | 2217/3750 [00:03<00:02, 713.27it/s][A
    batches:  61%|██████    | 2289/3750 [00:03<00:02, 688.12it/s][A
    batches:  63%|██████▎   | 2359/3750 [00:03<00:02, 675.05it/s][A
    batches:  65%|██████▍   | 2427/3750 [00:03<00:01, 673.53it/s][A
    batches:  67%|██████▋   | 2495/3750 [00:03<00:01, 673.41it/s][A
    batches:  68%|██████▊   | 2563/3750 [00:03<00:01, 672.79it/s][A
    batches:  70%|███████   | 2635/3750 [00:03<00:01, 686.10it/s][A
    batches:  72%|███████▏  | 2710/3750 [00:03<00:01, 704.00it/s][A
    batches:  74%|███████▍  | 2783/3750 [00:03<00:01, 710.88it/s][A
    batches:  76%|███████▌  | 2856/3750 [00:04<00:01, 714.86it/s][A
    batches:  78%|███████▊  | 2929/3750 [00:04<00:01, 719.28it/s][A
    batches:  80%|████████  | 3001/3750 [00:04<00:01, 715.40it/s][A
    batches:  82%|████████▏ | 3075/3750 [00:04<00:00, 720.03it/s][A
    batches:  84%|████████▍ | 3148/3750 [00:04<00:00, 717.36it/s][A
    batches:  86%|████████▌ | 3220/3750 [00:04<00:00, 709.27it/s][A
    batches:  88%|████████▊ | 3296/3750 [00:04<00:00, 721.13it/s][A
    batches:  90%|████████▉ | 3372/3750 [00:04<00:00, 732.43it/s][A
    batches:  92%|█████████▏| 3448/3750 [00:04<00:00, 738.81it/s][A
    batches:  94%|█████████▍| 3524/3750 [00:04<00:00, 742.54it/s][A
    batches:  96%|█████████▌| 3600/3750 [00:05<00:00, 745.29it/s][A
    batches:  98%|█████████▊| 3675/3750 [00:05<00:00, 746.49it/s][A
    epochs:  43%|████▎     | 13/30 [01:51<01:31,  5.37s/it]      [A
    batches:   0%|          | 0/3750 [00:00<?, ?it/s][A
    batches:   2%|▏         | 58/3750 [00:00<00:06, 573.41it/s][A
    batches:   4%|▎         | 134/3750 [00:00<00:05, 673.90it/s][A
    batches:   6%|▌         | 209/3750 [00:00<00:05, 707.74it/s][A
    batches:   7%|▋         | 280/3750 [00:00<00:05, 679.83it/s][A
    batches:   9%|▉         | 352/3750 [00:00<00:04, 692.15it/s][A
    batches:  11%|█▏        | 422/3750 [00:00<00:04, 681.04it/s][A
    batches:  13%|█▎        | 494/3750 [00:00<00:04, 691.38it/s][A
    batches:  15%|█▌        | 566/3750 [00:00<00:04, 699.65it/s][A
    batches:  17%|█▋        | 637/3750 [00:00<00:04, 695.06it/s][A
    batches:  19%|█▉        | 707/3750 [00:01<00:04, 671.33it/s][A
    batches:  21%|██        | 779/3750 [00:01<00:04, 685.30it/s][A
    batches:  23%|██▎       | 850/3750 [00:01<00:04, 691.70it/s][A
    batches:  25%|██▍       | 922/3750 [00:01<00:04, 699.59it/s][A
    batches:  26%|██▋       | 993/3750 [00:01<00:03, 701.16it/s][A
    batches:  28%|██▊       | 1066/3750 [00:01<00:03, 708.80it/s][A
    batches:  30%|███       | 1139/3750 [00:01<00:03, 714.09it/s][A
    batches:  32%|███▏      | 1212/3750 [00:01<00:03, 718.67it/s][A
    batches:  34%|███▍      | 1285/3750 [00:01<00:03, 720.67it/s][A
    batches:  36%|███▌      | 1358/3750 [00:01<00:03, 723.12it/s][A
    batches:  38%|███▊      | 1431/3750 [00:02<00:03, 724.42it/s][A
    batches:  40%|████      | 1504/3750 [00:02<00:03, 725.27it/s][A
    batches:  42%|████▏     | 1577/3750 [00:02<00:02, 726.31it/s][A
    batches:  44%|████▍     | 1650/3750 [00:02<00:02, 721.99it/s][A
    batches:  46%|████▌     | 1724/3750 [00:02<00:02, 724.56it/s][A
    batches:  48%|████▊     | 1797/3750 [00:02<00:02, 723.21it/s][A
    batches:  50%|████▉     | 1870/3750 [00:02<00:02, 717.66it/s][A
    batches:  52%|█████▏    | 1945/3750 [00:02<00:02, 724.63it/s][A
    batches:  54%|█████▍    | 2020/3750 [00:02<00:02, 731.71it/s][A
    batches:  56%|█████▌    | 2094/3750 [00:02<00:02, 729.75it/s][A
    batches:  58%|█████▊    | 2170/3750 [00:03<00:02, 737.33it/s][A
    batches:  60%|█████▉    | 2246/3750 [00:03<00:02, 742.99it/s][A
    batches:  62%|██████▏   | 2322/3750 [00:03<00:01, 745.92it/s][A
    batches:  64%|██████▍   | 2397/3750 [00:03<00:01, 745.78it/s][A
    batches:  66%|██████▌   | 2473/3750 [00:03<00:01, 748.66it/s][A
    batches:  68%|██████▊   | 2549/3750 [00:03<00:01, 749.62it/s][A
    batches:  70%|███████   | 2625/3750 [00:03<00:01, 750.69it/s][A
    batches:  72%|███████▏  | 2701/3750 [00:03<00:01, 749.68it/s][A
    batches:  74%|███████▍  | 2777/3750 [00:03<00:01, 751.37it/s][A
    batches:  76%|███████▌  | 2853/3750 [00:03<00:01, 753.43it/s][A
    batches:  78%|███████▊  | 2929/3750 [00:04<00:01, 753.36it/s][A
    batches:  80%|████████  | 3005/3750 [00:04<00:00, 754.55it/s][A
    batches:  82%|████████▏ | 3081/3750 [00:04<00:00, 754.13it/s][A
    batches:  84%|████████▍ | 3157/3750 [00:04<00:00, 752.03it/s][A
    batches:  86%|████████▌ | 3233/3750 [00:04<00:00, 753.01it/s][A
    batches:  88%|████████▊ | 3309/3750 [00:04<00:00, 752.00it/s][A
    batches:  90%|█████████ | 3385/3750 [00:04<00:00, 750.44it/s][A
    batches:  92%|█████████▏| 3461/3750 [00:04<00:00, 750.72it/s][A
    batches:  94%|█████████▍| 3537/3750 [00:04<00:00, 751.45it/s][A
    batches:  96%|█████████▋| 3613/3750 [00:04<00:00, 752.39it/s][A
    batches:  98%|█████████▊| 3689/3750 [00:05<00:00, 750.60it/s][A
    epochs:  47%|████▋     | 14/30 [01:56<01:24,  5.30s/it]      [A
    batches:   0%|          | 0/3750 [00:00<?, ?it/s][A
    batches:   2%|▏         | 59/3750 [00:00<00:06, 588.54it/s][A
    batches:   3%|▎         | 131/3750 [00:00<00:05, 663.83it/s][A
    batches:   5%|▌         | 204/3750 [00:00<00:05, 693.71it/s][A
    batches:   7%|▋         | 274/3750 [00:00<00:05, 691.35it/s][A
    batches:   9%|▉         | 344/3750 [00:00<00:05, 674.67it/s][A
    batches:  11%|█         | 418/3750 [00:00<00:04, 696.18it/s][A
    batches:  13%|█▎        | 488/3750 [00:00<00:04, 696.66it/s][A
    batches:  15%|█▍        | 558/3750 [00:00<00:04, 663.66it/s][A
    batches:  17%|█▋        | 631/3750 [00:00<00:04, 682.45it/s][A
    batches:  19%|█▉        | 705/3750 [00:01<00:04, 698.71it/s][A
    batches:  21%|██        | 780/3750 [00:01<00:04, 711.81it/s][A
    batches:  23%|██▎       | 853/3750 [00:01<00:04, 715.24it/s][A
    batches:  25%|██▍       | 928/3750 [00:01<00:03, 723.44it/s][A
    batches:  27%|██▋       | 1003/3750 [00:01<00:03, 729.98it/s][A
    batches:  29%|██▊       | 1077/3750 [00:01<00:03, 703.28it/s][A
    batches:  31%|███       | 1148/3750 [00:01<00:03, 682.35it/s][A
    batches:  32%|███▏      | 1217/3750 [00:01<00:03, 665.69it/s][A
    batches:  34%|███▍      | 1285/3750 [00:01<00:03, 667.75it/s][A
    batches:  36%|███▌      | 1352/3750 [00:01<00:03, 667.52it/s][A
    batches:  38%|███▊      | 1421/3750 [00:02<00:03, 672.06it/s][A
    batches:  40%|███▉      | 1493/3750 [00:02<00:03, 683.55it/s][A
    batches:  42%|████▏     | 1566/3750 [00:02<00:03, 697.17it/s][A
    batches:  44%|████▎     | 1639/3750 [00:02<00:02, 704.30it/s][A
    batches:  46%|████▌     | 1712/3750 [00:02<00:02, 710.08it/s][A
    batches:  48%|████▊     | 1784/3750 [00:02<00:02, 711.94it/s][A
    batches:  49%|████▉     | 1856/3750 [00:02<00:02, 706.92it/s][A
    batches:  51%|█████▏    | 1929/3750 [00:02<00:02, 712.52it/s][A
    batches:  53%|█████▎    | 2003/3750 [00:02<00:02, 718.08it/s][A
    batches:  55%|█████▌    | 2075/3750 [00:02<00:02, 710.52it/s][A
    batches:  57%|█████▋    | 2149/3750 [00:03<00:02, 718.97it/s][A
    batches:  59%|█████▉    | 2223/3750 [00:03<00:02, 724.91it/s][A
    batches:  61%|██████▏   | 2297/3750 [00:03<00:01, 726.54it/s][A
    batches:  63%|██████▎   | 2372/3750 [00:03<00:01, 730.54it/s][A
    batches:  65%|██████▌   | 2446/3750 [00:03<00:01, 733.29it/s][A
    batches:  67%|██████▋   | 2520/3750 [00:03<00:01, 733.50it/s][A
    batches:  69%|██████▉   | 2594/3750 [00:03<00:01, 734.52it/s][A
    batches:  71%|███████   | 2668/3750 [00:03<00:01, 735.63it/s][A
    batches:  73%|███████▎  | 2742/3750 [00:03<00:01, 736.08it/s][A
    batches:  75%|███████▌  | 2816/3750 [00:03<00:01, 737.08it/s][A
    batches:  77%|███████▋  | 2890/3750 [00:04<00:01, 736.82it/s][A
    batches:  79%|███████▉  | 2964/3750 [00:04<00:01, 737.37it/s][A
    batches:  81%|████████  | 3038/3750 [00:04<00:00, 736.58it/s][A
    batches:  83%|████████▎ | 3113/3750 [00:04<00:00, 738.53it/s][A
    batches:  85%|████████▌ | 3188/3750 [00:04<00:00, 740.43it/s][A
    batches:  87%|████████▋ | 3263/3750 [00:04<00:00, 740.73it/s][A
    batches:  89%|████████▉ | 3338/3750 [00:04<00:00, 739.63it/s][A
    batches:  91%|█████████ | 3413/3750 [00:04<00:00, 741.32it/s][A
    batches:  93%|█████████▎| 3488/3750 [00:04<00:00, 741.93it/s][A
    batches:  95%|█████████▌| 3563/3750 [00:04<00:00, 743.44it/s][A
    batches:  97%|█████████▋| 3638/3750 [00:05<00:00, 743.31it/s][A
    batches:  99%|█████████▉| 3713/3750 [00:05<00:00, 738.41it/s][A
    epochs:  50%|█████     | 15/30 [02:02<01:19,  5.29s/it]      [A
    batches:   0%|          | 0/3750 [00:00<?, ?it/s][A
    batches:   1%|▏         | 56/3750 [00:00<00:06, 554.91it/s][A
    batches:   3%|▎         | 130/3750 [00:00<00:05, 658.78it/s][A
    batches:   5%|▌         | 204/3750 [00:00<00:05, 694.47it/s][A
    batches:   7%|▋         | 279/3750 [00:00<00:04, 713.87it/s][A
    batches:   9%|▉         | 354/3750 [00:00<00:04, 724.93it/s][A
    batches:  11%|█▏        | 429/3750 [00:00<00:04, 732.54it/s][A
    batches:  13%|█▎        | 504/3750 [00:00<00:04, 736.47it/s][A
    batches:  15%|█▌        | 579/3750 [00:00<00:04, 739.21it/s][A
    batches:  17%|█▋        | 654/3750 [00:00<00:04, 740.06it/s][A
    batches:  19%|█▉        | 729/3750 [00:01<00:04, 740.82it/s][A
    batches:  21%|██▏       | 804/3750 [00:01<00:03, 742.43it/s][A
    batches:  23%|██▎       | 879/3750 [00:01<00:03, 739.55it/s][A
    batches:  25%|██▌       | 956/3750 [00:01<00:03, 746.47it/s][A
    batches:  27%|██▋       | 1031/3750 [00:01<00:03, 744.31it/s][A
    batches:  29%|██▉       | 1106/3750 [00:01<00:03, 745.78it/s][A
    batches:  32%|███▏      | 1183/3750 [00:01<00:03, 751.58it/s][A
    batches:  34%|███▎      | 1259/3750 [00:01<00:03, 751.75it/s][A
    batches:  36%|███▌      | 1336/3750 [00:01<00:03, 754.78it/s][A
    batches:  38%|███▊      | 1412/3750 [00:01<00:03, 754.51it/s][A
    batches:  40%|███▉      | 1488/3750 [00:02<00:03, 751.21it/s][A
    batches:  42%|████▏     | 1564/3750 [00:02<00:02, 751.64it/s][A
    batches:  44%|████▍     | 1641/3750 [00:02<00:02, 754.40it/s][A
    batches:  46%|████▌     | 1717/3750 [00:02<00:02, 750.51it/s][A
    batches:  48%|████▊     | 1793/3750 [00:02<00:02, 752.92it/s][A
    batches:  50%|████▉     | 1869/3750 [00:02<00:02, 753.81it/s][A
    batches:  52%|█████▏    | 1945/3750 [00:02<00:02, 754.75it/s][A
    batches:  54%|█████▍    | 2021/3750 [00:02<00:02, 754.70it/s][A
    batches:  56%|█████▌    | 2097/3750 [00:02<00:02, 754.08it/s][A
    batches:  58%|█████▊    | 2173/3750 [00:02<00:02, 747.16it/s][A
    batches:  60%|█████▉    | 2248/3750 [00:03<00:02, 726.34it/s][A
    batches:  62%|██████▏   | 2322/3750 [00:03<00:01, 728.59it/s][A
    batches:  64%|██████▍   | 2396/3750 [00:03<00:01, 729.23it/s][A
    batches:  66%|██████▌   | 2470/3750 [00:03<00:01, 730.03it/s][A
    batches:  68%|██████▊   | 2544/3750 [00:03<00:01, 728.15it/s][A
    batches:  70%|██████▉   | 2617/3750 [00:03<00:01, 725.65it/s][A
    batches:  72%|███████▏  | 2690/3750 [00:03<00:01, 726.46it/s][A
    batches:  74%|███████▎  | 2763/3750 [00:03<00:01, 727.45it/s][A
    batches:  76%|███████▌  | 2836/3750 [00:03<00:01, 726.09it/s][A
    batches:  78%|███████▊  | 2909/3750 [00:03<00:01, 719.08it/s][A
    batches:  80%|███████▉  | 2984/3750 [00:04<00:01, 725.72it/s][A
    batches:  82%|████████▏ | 3059/3750 [00:04<00:00, 732.68it/s][A
    batches:  84%|████████▎ | 3134/3750 [00:04<00:00, 736.29it/s][A
    batches:  86%|████████▌ | 3209/3750 [00:04<00:00, 738.76it/s][A
    batches:  88%|████████▊ | 3284/3750 [00:04<00:00, 739.33it/s][A
    batches:  90%|████████▉ | 3359/3750 [00:04<00:00, 741.15it/s][A
    batches:  92%|█████████▏| 3434/3750 [00:04<00:00, 743.07it/s][A
    batches:  94%|█████████▎| 3509/3750 [00:04<00:00, 742.65it/s][A
    batches:  96%|█████████▌| 3584/3750 [00:04<00:00, 741.89it/s][A
    batches:  98%|█████████▊| 3659/3750 [00:04<00:00, 743.31it/s][A
    batches: 100%|█████████▉| 3736/3750 [00:05<00:00, 749.31it/s][A
    epochs:  53%|█████▎    | 16/30 [02:07<01:13,  5.23s/it]      [A
    batches:   0%|          | 0/3750 [00:00<?, ?it/s][A
    batches:   1%|▏         | 51/3750 [00:00<00:07, 505.27it/s][A
    batches:   3%|▎         | 117/3750 [00:00<00:06, 590.98it/s][A
    batches:   5%|▌         | 188/3750 [00:00<00:05, 637.32it/s][A
    batches:   7%|▋         | 258/3750 [00:00<00:05, 660.68it/s][A
    batches:   9%|▉         | 331/3750 [00:00<00:04, 683.89it/s][A
    batches:  11%|█         | 401/3750 [00:00<00:04, 688.27it/s][A
    batches:  13%|█▎        | 470/3750 [00:00<00:04, 672.53it/s][A
    batches:  14%|█▍        | 543/3750 [00:00<00:04, 690.00it/s][A
    batches:  16%|█▋        | 615/3750 [00:00<00:04, 698.21it/s][A
    batches:  18%|█▊        | 689/3750 [00:01<00:04, 709.17it/s][A
    batches:  20%|██        | 763/3750 [00:01<00:04, 700.08it/s][A
    batches:  22%|██▏       | 834/3750 [00:01<00:04, 682.35it/s][A
    batches:  24%|██▍       | 904/3750 [00:01<00:04, 686.53it/s][A
    batches:  26%|██▌       | 973/3750 [00:01<00:04, 679.86it/s][A
    batches:  28%|██▊       | 1042/3750 [00:01<00:04, 645.16it/s][A
    batches:  30%|██▉       | 1113/3750 [00:01<00:03, 662.66it/s][A
    batches:  32%|███▏      | 1185/3750 [00:01<00:03, 677.21it/s][A
    batches:  33%|███▎      | 1256/3750 [00:01<00:03, 685.24it/s][A
    batches:  35%|███▌      | 1331/3750 [00:01<00:03, 703.71it/s][A
    batches:  37%|███▋      | 1406/3750 [00:02<00:03, 715.50it/s][A
    batches:  39%|███▉      | 1478/3750 [00:02<00:03, 691.12it/s][A
    batches:  41%|████▏     | 1548/3750 [00:02<00:03, 681.53it/s][A
    batches:  43%|████▎     | 1617/3750 [00:02<00:03, 668.22it/s][A
    batches:  45%|████▍     | 1684/3750 [00:02<00:03, 667.57it/s][A
    batches:  47%|████▋     | 1751/3750 [00:02<00:03, 663.06it/s][A
    batches:  49%|████▊     | 1823/3750 [00:02<00:02, 676.70it/s][A
    batches:  51%|█████     | 1895/3750 [00:02<00:02, 687.05it/s][A
    batches:  52%|█████▏    | 1965/3750 [00:02<00:02, 690.17it/s][A
    batches:  54%|█████▍    | 2036/3750 [00:02<00:02, 695.77it/s][A
    batches:  56%|█████▌    | 2107/3750 [00:03<00:02, 698.91it/s][A
    batches:  58%|█████▊    | 2180/3750 [00:03<00:02, 705.73it/s][A
    batches:  60%|██████    | 2251/3750 [00:03<00:02, 705.90it/s][A
    batches:  62%|██████▏   | 2324/3750 [00:03<00:02, 711.15it/s][A
    batches:  64%|██████▍   | 2396/3750 [00:03<00:01, 708.08it/s][A
    batches:  66%|██████▌   | 2472/3750 [00:03<00:01, 721.14it/s][A
    batches:  68%|██████▊   | 2548/3750 [00:03<00:01, 730.16it/s][A
    batches:  70%|██████▉   | 2622/3750 [00:03<00:01, 723.89it/s][A
    batches:  72%|███████▏  | 2696/3750 [00:03<00:01, 727.27it/s][A
    batches:  74%|███████▍  | 2770/3750 [00:04<00:01, 730.62it/s][A
    batches:  76%|███████▌  | 2844/3750 [00:04<00:01, 733.16it/s][A
    batches:  78%|███████▊  | 2918/3750 [00:04<00:01, 734.80it/s][A
    batches:  80%|███████▉  | 2993/3750 [00:04<00:01, 737.46it/s][A
    batches:  82%|████████▏ | 3067/3750 [00:04<00:00, 737.63it/s][A
    batches:  84%|████████▍ | 3141/3750 [00:04<00:00, 732.60it/s][A
    batches:  86%|████████▌ | 3216/3750 [00:04<00:00, 735.89it/s][A
    batches:  88%|████████▊ | 3293/3750 [00:04<00:00, 743.74it/s][A
    batches:  90%|████████▉ | 3369/3750 [00:04<00:00, 747.28it/s][A
    batches:  92%|█████████▏| 3444/3750 [00:04<00:00, 743.97it/s][A
    batches:  94%|█████████▍| 3520/3750 [00:05<00:00, 748.59it/s][A
    batches:  96%|█████████▌| 3597/3750 [00:05<00:00, 752.60it/s][A
    batches:  98%|█████████▊| 3674/3750 [00:05<00:00, 754.78it/s][A
    epochs:  57%|█████▋    | 17/30 [02:12<01:08,  5.25s/it]      [A
    batches:   0%|          | 0/3750 [00:00<?, ?it/s][A
    batches:   2%|▏         | 61/3750 [00:00<00:06, 604.56it/s][A
    batches:   4%|▎         | 137/3750 [00:00<00:05, 692.54it/s][A
    batches:   6%|▌         | 208/3750 [00:00<00:05, 699.54it/s][A
    batches:   7%|▋         | 278/3750 [00:00<00:05, 686.59it/s][A
    batches:   9%|▉         | 351/3750 [00:00<00:04, 700.60it/s][A
    batches:  11%|█▏        | 423/3750 [00:00<00:04, 705.64it/s][A
    batches:  13%|█▎        | 498/3750 [00:00<00:04, 717.74it/s][A
    batches:  15%|█▌        | 575/3750 [00:00<00:04, 732.35it/s][A
    batches:  17%|█▋        | 650/3750 [00:00<00:04, 735.34it/s][A
    batches:  19%|█▉        | 726/3750 [00:01<00:04, 742.83it/s][A
    batches:  21%|██▏       | 802/3750 [00:01<00:03, 746.09it/s][A
    batches:  23%|██▎       | 879/3750 [00:01<00:03, 750.93it/s][A
    batches:  25%|██▌       | 956/3750 [00:01<00:03, 755.00it/s][A
    batches:  28%|██▊       | 1032/3750 [00:01<00:03, 755.42it/s][A
    batches:  30%|██▉       | 1109/3750 [00:01<00:03, 757.70it/s][A
    batches:  32%|███▏      | 1185/3750 [00:01<00:03, 739.56it/s][A
    batches:  34%|███▎      | 1264/3750 [00:01<00:03, 752.16it/s][A
    batches:  36%|███▌      | 1342/3750 [00:01<00:03, 758.49it/s][A
    batches:  38%|███▊      | 1420/3750 [00:01<00:03, 763.24it/s][A
    batches:  40%|███▉      | 1497/3750 [00:02<00:02, 760.90it/s][A
    batches:  42%|████▏     | 1574/3750 [00:02<00:02, 762.71it/s][A
    batches:  44%|████▍     | 1652/3750 [00:02<00:02, 766.20it/s][A
    batches:  46%|████▌     | 1730/3750 [00:02<00:02, 768.27it/s][A
    batches:  48%|████▊     | 1807/3750 [00:02<00:02, 759.95it/s][A
    batches:  50%|█████     | 1884/3750 [00:02<00:02, 756.45it/s][A
    batches:  52%|█████▏    | 1960/3750 [00:02<00:02, 751.23it/s][A
    batches:  54%|█████▍    | 2036/3750 [00:02<00:02, 746.94it/s][A
    batches:  56%|█████▋    | 2111/3750 [00:02<00:02, 742.01it/s][A
    batches:  58%|█████▊    | 2186/3750 [00:02<00:02, 734.87it/s][A
    batches:  60%|██████    | 2260/3750 [00:03<00:02, 733.13it/s][A
    batches:  62%|██████▏   | 2334/3750 [00:03<00:01, 731.37it/s][A
    batches:  64%|██████▍   | 2408/3750 [00:03<00:01, 729.29it/s][A
    batches:  66%|██████▌   | 2481/3750 [00:03<00:01, 728.34it/s][A
    batches:  68%|██████▊   | 2554/3750 [00:03<00:01, 728.27it/s][A
    batches:  70%|███████   | 2627/3750 [00:03<00:01, 722.71it/s][A
    batches:  72%|███████▏  | 2700/3750 [00:03<00:01, 722.60it/s][A
    batches:  74%|███████▍  | 2773/3750 [00:03<00:01, 722.50it/s][A
    batches:  76%|███████▌  | 2846/3750 [00:03<00:01, 723.25it/s][A
    batches:  78%|███████▊  | 2919/3750 [00:03<00:01, 722.69it/s][A
    batches:  80%|███████▉  | 2992/3750 [00:04<00:01, 722.80it/s][A
    batches:  82%|████████▏ | 3065/3750 [00:04<00:00, 721.86it/s][A
    batches:  84%|████████▎ | 3138/3750 [00:04<00:00, 722.94it/s][A
    batches:  86%|████████▌ | 3211/3750 [00:04<00:00, 723.40it/s][A
    batches:  88%|████████▊ | 3284/3750 [00:04<00:00, 723.38it/s][A
    batches:  90%|████████▉ | 3357/3750 [00:04<00:00, 724.59it/s][A
    batches:  91%|█████████▏| 3430/3750 [00:04<00:00, 724.17it/s][A
    batches:  93%|█████████▎| 3503/3750 [00:04<00:00, 722.26it/s][A
    batches:  95%|█████████▌| 3576/3750 [00:04<00:00, 702.25it/s][A
    batches:  97%|█████████▋| 3649/3750 [00:04<00:00, 707.87it/s][A
    batches:  99%|█████████▉| 3720/3750 [00:05<00:00, 705.75it/s][A
    epochs:  60%|██████    | 18/30 [02:17<01:02,  5.21s/it]      [A
    batches:   0%|          | 0/3750 [00:00<?, ?it/s][A
    batches:   1%|▏         | 52/3750 [00:00<00:07, 514.21it/s][A
    batches:   3%|▎         | 122/3750 [00:00<00:05, 618.99it/s][A
    batches:   5%|▌         | 195/3750 [00:00<00:05, 665.18it/s][A
    batches:   7%|▋         | 268/3750 [00:00<00:05, 682.31it/s][A
    batches:   9%|▉         | 341/3750 [00:00<00:05, 676.09it/s][A
    batches:  11%|█         | 411/3750 [00:00<00:04, 682.42it/s][A
    batches:  13%|█▎        | 484/3750 [00:00<00:04, 696.07it/s][A
    batches:  15%|█▍        | 554/3750 [00:00<00:04, 696.98it/s][A
    batches:  17%|█▋        | 624/3750 [00:00<00:04, 664.83it/s][A
    batches:  19%|█▊        | 696/3750 [00:01<00:04, 679.84it/s][A
    batches:  21%|██        | 770/3750 [00:01<00:04, 696.83it/s][A
    batches:  22%|██▏       | 842/3750 [00:01<00:04, 703.49it/s][A
    batches:  24%|██▍       | 916/3750 [00:01<00:03, 713.71it/s][A
    batches:  26%|██▋       | 991/3750 [00:01<00:03, 722.08it/s][A
    batches:  28%|██▊       | 1064/3750 [00:01<00:03, 723.70it/s][A
    batches:  30%|███       | 1137/3750 [00:01<00:03, 704.14it/s][A
    batches:  32%|███▏      | 1208/3750 [00:01<00:03, 681.38it/s][A
    batches:  34%|███▍      | 1277/3750 [00:01<00:03, 668.09it/s][A
    batches:  36%|███▌      | 1344/3750 [00:01<00:03, 656.58it/s][A
    batches:  38%|███▊      | 1410/3750 [00:02<00:03, 653.98it/s][A
    batches:  39%|███▉      | 1476/3750 [00:02<00:03, 655.40it/s][A
    batches:  41%|████      | 1546/3750 [00:02<00:03, 667.51it/s][A
    batches:  43%|████▎     | 1615/3750 [00:02<00:03, 672.65it/s][A
    batches:  45%|████▍     | 1684/3750 [00:02<00:03, 676.06it/s][A
    batches:  47%|████▋     | 1761/3750 [00:02<00:02, 702.23it/s][A
    batches:  49%|████▉     | 1839/3750 [00:02<00:02, 720.89it/s][A
    batches:  51%|█████     | 1912/3750 [00:02<00:02, 707.95it/s][A
    batches:  53%|█████▎    | 1983/3750 [00:02<00:02, 697.46it/s][A
    batches:  55%|█████▍    | 2055/3750 [00:02<00:02, 703.76it/s][A
    batches:  57%|█████▋    | 2126/3750 [00:03<00:02, 690.79it/s][A
    batches:  59%|█████▊    | 2196/3750 [00:03<00:02, 692.72it/s][A
    batches:  60%|██████    | 2268/3750 [00:03<00:02, 698.50it/s][A
    batches:  62%|██████▏   | 2338/3750 [00:03<00:02, 685.35it/s][A
    batches:  64%|██████▍   | 2409/3750 [00:03<00:01, 691.69it/s][A
    batches:  66%|██████▌   | 2481/3750 [00:03<00:01, 696.95it/s][A
    batches:  68%|██████▊   | 2553/3750 [00:03<00:01, 702.81it/s][A
    batches:  70%|██████▉   | 2624/3750 [00:03<00:01, 692.64it/s][A
    batches:  72%|███████▏  | 2696/3750 [00:03<00:01, 699.51it/s][A
    batches:  74%|███████▍  | 2771/3750 [00:04<00:01, 712.78it/s][A
    batches:  76%|███████▌  | 2849/3750 [00:04<00:01, 730.60it/s][A
    batches:  78%|███████▊  | 2926/3750 [00:04<00:01, 719.60it/s][A
    batches:  80%|████████  | 3002/3750 [00:04<00:01, 730.31it/s][A
    batches:  82%|████████▏ | 3078/3750 [00:04<00:00, 736.80it/s][A
    batches:  84%|████████▍ | 3152/3750 [00:04<00:00, 730.35it/s][A
    batches:  86%|████████▌ | 3226/3750 [00:04<00:00, 728.71it/s][A
    batches:  88%|████████▊ | 3299/3750 [00:04<00:00, 704.27it/s][A
    batches:  90%|████████▉ | 3372/3750 [00:04<00:00, 710.25it/s][A
    batches:  92%|█████████▏| 3444/3750 [00:04<00:00, 706.52it/s][A
    batches:  94%|█████████▎| 3515/3750 [00:05<00:00, 670.98it/s][A
    batches:  96%|█████████▌| 3587/3750 [00:05<00:00, 684.82it/s][A
    batches:  98%|█████████▊| 3659/3750 [00:05<00:00, 692.46it/s][A
    batches: 100%|█████████▉| 3733/3750 [00:05<00:00, 704.93it/s][A
    epochs:  63%|██████▎   | 19/30 [02:23<00:57,  5.27s/it]      [A
    batches:   0%|          | 0/3750 [00:00<?, ?it/s][A
    batches:   2%|▏         | 61/3750 [00:00<00:06, 601.20it/s][A
    batches:   4%|▎         | 139/3750 [00:00<00:05, 700.47it/s][A
    batches:   6%|▌         | 216/3750 [00:00<00:04, 731.51it/s][A
    batches:   8%|▊         | 290/3750 [00:00<00:04, 695.00it/s][A
    batches:  10%|▉         | 360/3750 [00:00<00:04, 684.87it/s][A
    batches:  11%|█▏        | 429/3750 [00:00<00:04, 684.54it/s][A
    batches:  13%|█▎        | 500/3750 [00:00<00:04, 690.45it/s][A
    batches:  15%|█▌        | 572/3750 [00:00<00:04, 698.02it/s][A
    batches:  17%|█▋        | 648/3750 [00:00<00:04, 714.39it/s][A
    batches:  19%|█▉        | 723/3750 [00:01<00:04, 724.83it/s][A
    batches:  21%|██▏       | 800/3750 [00:01<00:04, 736.08it/s][A
    batches:  23%|██▎       | 878/3750 [00:01<00:03, 748.14it/s][A
    batches:  25%|██▌       | 955/3750 [00:01<00:03, 752.37it/s][A
    batches:  27%|██▋       | 1031/3750 [00:01<00:03, 745.79it/s][A
    batches:  30%|██▉       | 1108/3750 [00:01<00:03, 751.37it/s][A
    batches:  32%|███▏      | 1184/3750 [00:01<00:03, 749.87it/s][A
    batches:  34%|███▎      | 1262/3750 [00:01<00:03, 757.45it/s][A
    batches:  36%|███▌      | 1341/3750 [00:01<00:03, 766.06it/s][A
    batches:  38%|███▊      | 1418/3750 [00:01<00:03, 747.80it/s][A
    batches:  40%|███▉      | 1497/3750 [00:02<00:02, 757.92it/s][A
    batches:  42%|████▏     | 1576/3750 [00:02<00:02, 765.01it/s][A
    batches:  44%|████▍     | 1655/3750 [00:02<00:02, 771.87it/s][A
    batches:  46%|████▌     | 1733/3750 [00:02<00:02, 759.18it/s][A
    batches:  48%|████▊     | 1811/3750 [00:02<00:02, 765.24it/s][A
    batches:  50%|█████     | 1888/3750 [00:02<00:02, 762.18it/s][A
    batches:  52%|█████▏    | 1966/3750 [00:02<00:02, 766.29it/s][A
    batches:  54%|█████▍    | 2043/3750 [00:02<00:02, 736.84it/s][A
    batches:  57%|█████▋    | 2122/3750 [00:02<00:02, 750.08it/s][A
    batches:  59%|█████▊    | 2201/3750 [00:02<00:02, 759.22it/s][A
    batches:  61%|██████    | 2278/3750 [00:03<00:01, 757.63it/s][A
    batches:  63%|██████▎   | 2356/3750 [00:03<00:01, 762.86it/s][A
    batches:  65%|██████▍   | 2433/3750 [00:03<00:01, 738.54it/s][A
    batches:  67%|██████▋   | 2510/3750 [00:03<00:01, 744.72it/s][A
    batches:  69%|██████▉   | 2588/3750 [00:03<00:01, 753.50it/s][A
    batches:  71%|███████   | 2666/3750 [00:03<00:01, 760.90it/s][A
    batches:  73%|███████▎  | 2743/3750 [00:03<00:01, 757.33it/s][A
    batches:  75%|███████▌  | 2822/3750 [00:03<00:01, 764.43it/s][A
    batches:  77%|███████▋  | 2899/3750 [00:03<00:01, 720.66it/s][A
    batches:  79%|███████▉  | 2977/3750 [00:04<00:01, 734.92it/s][A
    batches:  81%|████████▏ | 3056/3750 [00:04<00:00, 748.76it/s][A
    batches:  84%|████████▎ | 3134/3750 [00:04<00:00, 756.75it/s][A
    batches:  86%|████████▌ | 3212/3750 [00:04<00:00, 761.59it/s][A
    batches:  88%|████████▊ | 3290/3750 [00:04<00:00, 766.99it/s][A
    batches:  90%|████████▉ | 3369/3750 [00:04<00:00, 771.23it/s][A
    batches:  92%|█████████▏| 3447/3750 [00:04<00:00, 773.52it/s][A
    batches:  94%|█████████▍| 3525/3750 [00:04<00:00, 750.06it/s][A
    batches:  96%|█████████▌| 3601/3750 [00:04<00:00, 722.11it/s][A
    batches:  98%|█████████▊| 3674/3750 [00:04<00:00, 714.30it/s][A
    batches: 100%|█████████▉| 3746/3750 [00:05<00:00, 715.28it/s][A
    epochs:  67%|██████▋   | 20/30 [02:28<00:52,  5.21s/it]      [A
    batches:   0%|          | 0/3750 [00:00<?, ?it/s][A
    batches:   2%|▏         | 57/3750 [00:00<00:06, 550.07it/s][A
    batches:   4%|▎         | 134/3750 [00:00<00:05, 675.58it/s][A
    batches:   6%|▌         | 211/3750 [00:00<00:04, 714.83it/s][A
    batches:   8%|▊         | 284/3750 [00:00<00:04, 719.17it/s][A
    batches:  10%|▉         | 362/3750 [00:00<00:04, 737.65it/s][A
    batches:  12%|█▏        | 440/3750 [00:00<00:04, 749.96it/s][A
    batches:  14%|█▍        | 518/3750 [00:00<00:04, 757.94it/s][A
    batches:  16%|█▌        | 596/3750 [00:00<00:04, 762.70it/s][A
    batches:  18%|█▊        | 673/3750 [00:00<00:04, 747.71it/s][A
    batches:  20%|█▉        | 748/3750 [00:01<00:04, 735.90it/s][A
    batches:  22%|██▏       | 824/3750 [00:01<00:03, 741.38it/s][A
    batches:  24%|██▍       | 901/3750 [00:01<00:03, 748.35it/s][A
    batches:  26%|██▌       | 978/3750 [00:01<00:03, 753.47it/s][A
    batches:  28%|██▊       | 1054/3750 [00:01<00:03, 740.68it/s][A
    batches:  30%|███       | 1131/3750 [00:01<00:03, 747.41it/s][A
    batches:  32%|███▏      | 1208/3750 [00:01<00:03, 753.89it/s][A
    batches:  34%|███▍      | 1284/3750 [00:01<00:03, 741.54it/s][A
    batches:  36%|███▌      | 1359/3750 [00:01<00:03, 732.04it/s][A
    batches:  38%|███▊      | 1433/3750 [00:01<00:03, 716.05it/s][A
    batches:  40%|████      | 1505/3750 [00:02<00:03, 697.43it/s][A
    batches:  42%|████▏     | 1575/3750 [00:02<00:03, 688.19it/s][A
    batches:  44%|████▍     | 1648/3750 [00:02<00:03, 698.77it/s][A
    batches:  46%|████▌     | 1721/3750 [00:02<00:02, 707.23it/s][A
    batches:  48%|████▊     | 1793/3750 [00:02<00:02, 710.75it/s][A
    batches:  50%|████▉     | 1867/3750 [00:02<00:02, 716.95it/s][A
    batches:  52%|█████▏    | 1940/3750 [00:02<00:02, 719.68it/s][A
    batches:  54%|█████▎    | 2013/3750 [00:02<00:02, 721.19it/s][A
    batches:  56%|█████▌    | 2086/3750 [00:02<00:02, 713.06it/s][A
    batches:  58%|█████▊    | 2158/3750 [00:02<00:02, 695.18it/s][A
    batches:  59%|█████▉    | 2228/3750 [00:03<00:02, 686.01it/s][A
    batches:  61%|██████▏   | 2299/3750 [00:03<00:02, 692.77it/s][A
    batches:  63%|██████▎   | 2371/3750 [00:03<00:01, 697.37it/s][A
    batches:  65%|██████▌   | 2443/3750 [00:03<00:01, 702.46it/s][A
    batches:  67%|██████▋   | 2514/3750 [00:03<00:01, 699.43it/s][A
    batches:  69%|██████▉   | 2585/3750 [00:03<00:01, 701.73it/s][A
    batches:  71%|███████   | 2656/3750 [00:03<00:01, 702.24it/s][A
    batches:  73%|███████▎  | 2728/3750 [00:03<00:01, 704.95it/s][A
    batches:  75%|███████▍  | 2799/3750 [00:03<00:01, 704.09it/s][A
    batches:  77%|███████▋  | 2870/3750 [00:04<00:01, 702.13it/s][A
    batches:  78%|███████▊  | 2942/3750 [00:04<00:01, 705.45it/s][A
    batches:  80%|████████  | 3014/3750 [00:04<00:01, 709.47it/s][A
    batches:  82%|████████▏ | 3086/3750 [00:04<00:00, 712.21it/s][A
    batches:  84%|████████▍ | 3158/3750 [00:04<00:00, 701.98it/s][A
    batches:  86%|████████▌ | 3230/3750 [00:04<00:00, 705.46it/s][A
    batches:  88%|████████▊ | 3303/3750 [00:04<00:00, 709.47it/s][A
    batches:  90%|█████████ | 3376/3750 [00:04<00:00, 696.88it/s][A
    batches:  92%|█████████▏| 3446/3750 [00:04<00:00, 696.02it/s][A
    batches:  94%|█████████▍| 3518/3750 [00:04<00:00, 701.45it/s][A
    batches:  96%|█████████▌| 3596/3750 [00:05<00:00, 722.65it/s][A
    batches:  98%|█████████▊| 3675/3750 [00:05<00:00, 740.24it/s][A
    epochs:  70%|███████   | 21/30 [02:33<00:46,  5.21s/it]      [A
    batches:   0%|          | 0/3750 [00:00<?, ?it/s][A
    batches:   2%|▏         | 61/3750 [00:00<00:06, 609.42it/s][A
    batches:   4%|▎         | 138/3750 [00:00<00:05, 700.48it/s][A
    batches:   6%|▌         | 215/3750 [00:00<00:04, 730.96it/s][A
    batches:   8%|▊         | 292/3750 [00:00<00:04, 746.23it/s][A
    batches:  10%|▉         | 370/3750 [00:00<00:04, 757.48it/s][A
    batches:  12%|█▏        | 446/3750 [00:00<00:04, 758.26it/s][A
    batches:  14%|█▍        | 523/3750 [00:00<00:04, 761.94it/s][A
    batches:  16%|█▌        | 600/3750 [00:00<00:04, 759.43it/s][A
    batches:  18%|█▊        | 676/3750 [00:00<00:04, 747.07it/s][A
    batches:  20%|██        | 754/3750 [00:01<00:03, 754.72it/s][A
    batches:  22%|██▏       | 832/3750 [00:01<00:03, 759.91it/s][A
    batches:  24%|██▍       | 910/3750 [00:01<00:03, 764.54it/s][A
    batches:  26%|██▋       | 988/3750 [00:01<00:03, 768.99it/s][A
    batches:  28%|██▊       | 1066/3750 [00:01<00:03, 770.87it/s][A
    batches:  31%|███       | 1144/3750 [00:01<00:03, 772.61it/s][A
    batches:  33%|███▎      | 1222/3750 [00:01<00:03, 761.64it/s][A
    batches:  35%|███▍      | 1299/3750 [00:01<00:03, 735.04it/s][A
    batches:  37%|███▋      | 1373/3750 [00:01<00:03, 719.49it/s][A
    batches:  39%|███▊      | 1446/3750 [00:01<00:03, 713.49it/s][A
    batches:  41%|████      | 1521/3750 [00:02<00:03, 722.02it/s][A
    batches:  43%|████▎     | 1596/3750 [00:02<00:02, 728.30it/s][A
    batches:  45%|████▍     | 1670/3750 [00:02<00:02, 730.49it/s][A
    batches:  47%|████▋     | 1744/3750 [00:02<00:02, 720.90it/s][A
    batches:  48%|████▊     | 1817/3750 [00:02<00:02, 713.34it/s][A
    batches:  50%|█████     | 1889/3750 [00:02<00:02, 710.77it/s][A
    batches:  52%|█████▏    | 1961/3750 [00:02<00:02, 710.11it/s][A
    batches:  54%|█████▍    | 2033/3750 [00:02<00:02, 705.03it/s][A
    batches:  56%|█████▌    | 2105/3750 [00:02<00:02, 707.41it/s][A
    batches:  58%|█████▊    | 2176/3750 [00:02<00:02, 705.57it/s][A
    batches:  60%|█████▉    | 2247/3750 [00:03<00:02, 705.92it/s][A
    batches:  62%|██████▏   | 2318/3750 [00:03<00:02, 705.60it/s][A
    batches:  64%|██████▎   | 2389/3750 [00:03<00:01, 696.81it/s][A
    batches:  66%|██████▌   | 2464/3750 [00:03<00:01, 710.44it/s][A
    batches:  68%|██████▊   | 2536/3750 [00:03<00:01, 698.32it/s][A
    batches:  70%|██████▉   | 2607/3750 [00:03<00:01, 701.25it/s][A
    batches:  71%|███████▏  | 2678/3750 [00:03<00:01, 695.62it/s][A
    batches:  73%|███████▎  | 2750/3750 [00:03<00:01, 701.49it/s][A
    batches:  75%|███████▌  | 2822/3750 [00:03<00:01, 704.93it/s][A
    batches:  77%|███████▋  | 2893/3750 [00:03<00:01, 691.98it/s][A
    batches:  79%|███████▉  | 2963/3750 [00:04<00:01, 685.57it/s][A
    batches:  81%|████████  | 3032/3750 [00:04<00:01, 665.82it/s][A
    batches:  83%|████████▎ | 3104/3750 [00:04<00:00, 681.22it/s][A
    batches:  85%|████████▍ | 3173/3750 [00:04<00:00, 673.50it/s][A
    batches:  87%|████████▋ | 3245/3750 [00:04<00:00, 682.84it/s][A
    batches:  88%|████████▊ | 3314/3750 [00:04<00:00, 678.90it/s][A
    batches:  90%|█████████ | 3386/3750 [00:04<00:00, 688.99it/s][A
    batches:  92%|█████████▏| 3456/3750 [00:04<00:00, 690.92it/s][A
    batches:  94%|█████████▍| 3526/3750 [00:04<00:00, 690.95it/s][A
    batches:  96%|█████████▌| 3596/3750 [00:05<00:00, 656.17it/s][A
    batches:  98%|█████████▊| 3668/3750 [00:05<00:00, 672.53it/s][A
    batches: 100%|█████████▉| 3744/3750 [00:05<00:00, 696.46it/s][A
    epochs:  73%|███████▎  | 22/30 [02:38<00:41,  5.22s/it]      [A
    batches:   0%|          | 0/3750 [00:00<?, ?it/s][A
    batches:   1%|▏         | 53/3750 [00:00<00:07, 518.33it/s][A
    batches:   3%|▎         | 120/3750 [00:00<00:06, 604.26it/s][A
    batches:   5%|▌         | 196/3750 [00:00<00:05, 674.19it/s][A
    batches:   7%|▋         | 273/3750 [00:00<00:04, 711.04it/s][A
    batches:   9%|▉         | 349/3750 [00:00<00:04, 726.00it/s][A
    batches:  11%|█▏        | 422/3750 [00:00<00:04, 723.45it/s][A
    batches:  13%|█▎        | 495/3750 [00:00<00:04, 719.96it/s][A
    batches:  15%|█▌        | 568/3750 [00:00<00:04, 719.87it/s][A
    batches:  17%|█▋        | 640/3750 [00:00<00:04, 711.36it/s][A
    batches:  19%|█▉        | 712/3750 [00:01<00:04, 712.03it/s][A
    batches:  21%|██        | 785/3750 [00:01<00:04, 716.64it/s][A
    batches:  23%|██▎       | 860/3750 [00:01<00:03, 723.86it/s][A
    batches:  25%|██▍       | 937/3750 [00:01<00:03, 736.03it/s][A
    batches:  27%|██▋       | 1011/3750 [00:01<00:03, 733.31it/s][A
    batches:  29%|██▉       | 1088/3750 [00:01<00:03, 742.35it/s][A
    batches:  31%|███       | 1166/3750 [00:01<00:03, 751.98it/s][A
    batches:  33%|███▎      | 1242/3750 [00:01<00:03, 730.98it/s][A
    batches:  35%|███▌      | 1316/3750 [00:01<00:03, 727.39it/s][A
    batches:  37%|███▋      | 1389/3750 [00:01<00:03, 724.08it/s][A
    batches:  39%|███▉      | 1462/3750 [00:02<00:03, 721.19it/s][A
    batches:  41%|████      | 1535/3750 [00:02<00:03, 717.31it/s][A
    batches:  43%|████▎     | 1607/3750 [00:02<00:03, 711.77it/s][A
    batches:  45%|████▍     | 1684/3750 [00:02<00:02, 727.02it/s][A
    batches:  47%|████▋     | 1762/3750 [00:02<00:02, 742.52it/s][A
    batches:  49%|████▉     | 1840/3750 [00:02<00:02, 753.61it/s][A
    batches:  51%|█████     | 1918/3750 [00:02<00:02, 760.67it/s][A
    batches:  53%|█████▎    | 1996/3750 [00:02<00:02, 763.74it/s][A
    batches:  55%|█████▌    | 2075/3750 [00:02<00:02, 768.71it/s][A
    batches:  57%|█████▋    | 2152/3750 [00:02<00:02, 752.94it/s][A
    batches:  59%|█████▉    | 2228/3750 [00:03<00:02, 734.38it/s][A
    batches:  61%|██████▏   | 2302/3750 [00:03<00:02, 721.19it/s][A
    batches:  63%|██████▎   | 2375/3750 [00:03<00:01, 715.26it/s][A
    batches:  65%|██████▌   | 2447/3750 [00:03<00:01, 710.73it/s][A
    batches:  67%|██████▋   | 2519/3750 [00:03<00:01, 697.98it/s][A
    batches:  69%|██████▉   | 2589/3750 [00:03<00:01, 698.16it/s][A
    batches:  71%|███████   | 2659/3750 [00:03<00:01, 696.19it/s][A
    batches:  73%|███████▎  | 2729/3750 [00:03<00:01, 697.20it/s][A
    batches:  75%|███████▍  | 2799/3750 [00:03<00:01, 696.59it/s][A
    batches:  77%|███████▋  | 2869/3750 [00:03<00:01, 694.25it/s][A
    batches:  78%|███████▊  | 2939/3750 [00:04<00:01, 692.05it/s][A
    batches:  80%|████████  | 3009/3750 [00:04<00:01, 680.23it/s][A
    batches:  82%|████████▏ | 3079/3750 [00:04<00:00, 684.07it/s][A
    batches:  84%|████████▍ | 3148/3750 [00:04<00:00, 684.84it/s][A
    batches:  86%|████████▌ | 3217/3750 [00:04<00:00, 685.97it/s][A
    batches:  88%|████████▊ | 3286/3750 [00:04<00:00, 680.41it/s][A
    batches:  90%|████████▉ | 3357/3750 [00:04<00:00, 688.82it/s][A
    batches:  91%|█████████▏| 3427/3750 [00:04<00:00, 690.98it/s][A
    batches:  93%|█████████▎| 3498/3750 [00:04<00:00, 695.25it/s][A
    batches:  95%|█████████▌| 3569/3750 [00:05<00:00, 698.44it/s][A
    batches:  97%|█████████▋| 3640/3750 [00:05<00:00, 699.34it/s][A
    batches:  99%|█████████▉| 3713/3750 [00:05<00:00, 707.12it/s][A
    epochs:  77%|███████▋  | 23/30 [02:43<00:36,  5.23s/it]      [A
    batches:   0%|          | 0/3750 [00:00<?, ?it/s][A
    batches:   2%|▏         | 64/3750 [00:00<00:05, 638.10it/s][A
    batches:   4%|▎         | 138/3750 [00:00<00:05, 693.23it/s][A
    batches:   6%|▌         | 211/3750 [00:00<00:04, 709.39it/s][A
    batches:   8%|▊         | 288/3750 [00:00<00:04, 732.50it/s][A
    batches:  10%|▉         | 365/3750 [00:00<00:04, 744.49it/s][A
    batches:  12%|█▏        | 440/3750 [00:00<00:04, 722.94it/s][A
    batches:  14%|█▎        | 513/3750 [00:00<00:04, 692.45it/s][A
    batches:  16%|█▌        | 584/3750 [00:00<00:04, 696.44it/s][A
    batches:  17%|█▋        | 654/3750 [00:00<00:04, 686.09it/s][A
    batches:  19%|█▉        | 724/3750 [00:01<00:04, 688.74it/s][A
    batches:  21%|██        | 794/3750 [00:01<00:04, 691.75it/s][A
    batches:  23%|██▎       | 864/3750 [00:01<00:04, 681.16it/s][A
    batches:  25%|██▍       | 933/3750 [00:01<00:04, 660.25it/s][A
    batches:  27%|██▋       | 1007/3750 [00:01<00:04, 681.59it/s][A
    batches:  29%|██▉       | 1083/3750 [00:01<00:03, 701.89it/s][A
    batches:  31%|███       | 1158/3750 [00:01<00:03, 715.54it/s][A
    batches:  33%|███▎      | 1230/3750 [00:01<00:03, 694.50it/s][A
    batches:  35%|███▍      | 1306/3750 [00:01<00:03, 712.28it/s][A
    batches:  37%|███▋      | 1382/3750 [00:01<00:03, 724.26it/s][A
    batches:  39%|███▉      | 1455/3750 [00:02<00:03, 718.70it/s][A
    batches:  41%|████      | 1527/3750 [00:02<00:03, 711.07it/s][A
    batches:  43%|████▎     | 1599/3750 [00:02<00:03, 662.91it/s][A
    batches:  45%|████▍     | 1671/3750 [00:02<00:03, 677.72it/s][A
    batches:  47%|████▋     | 1747/3750 [00:02<00:02, 700.72it/s][A
    batches:  49%|████▊     | 1825/3750 [00:02<00:02, 722.56it/s][A
    batches:  51%|█████     | 1903/3750 [00:02<00:02, 739.04it/s][A
    batches:  53%|█████▎    | 1981/3750 [00:02<00:02, 749.58it/s][A
    batches:  55%|█████▍    | 2058/3750 [00:02<00:02, 753.85it/s][A
    batches:  57%|█████▋    | 2134/3750 [00:03<00:02, 742.73it/s][A
    batches:  59%|█████▉    | 2211/3750 [00:03<00:02, 750.03it/s][A
    batches:  61%|██████    | 2288/3750 [00:03<00:01, 755.45it/s][A
    batches:  63%|██████▎   | 2364/3750 [00:03<00:01, 731.79it/s][A
    batches:  65%|██████▌   | 2438/3750 [00:03<00:01, 721.10it/s][A
    batches:  67%|██████▋   | 2511/3750 [00:03<00:01, 715.08it/s][A
    batches:  69%|██████▉   | 2583/3750 [00:03<00:01, 709.95it/s][A
    batches:  71%|███████   | 2655/3750 [00:03<00:01, 708.87it/s][A
    batches:  73%|███████▎  | 2727/3750 [00:03<00:01, 709.57it/s][A
    batches:  75%|███████▍  | 2798/3750 [00:03<00:01, 705.87it/s][A
    batches:  77%|███████▋  | 2869/3750 [00:04<00:01, 701.65it/s][A
    batches:  78%|███████▊  | 2940/3750 [00:04<00:01, 691.77it/s][A
    batches:  80%|████████  | 3011/3750 [00:04<00:01, 694.87it/s][A
    batches:  82%|████████▏ | 3083/3750 [00:04<00:00, 699.74it/s][A
    batches:  84%|████████▍ | 3154/3750 [00:04<00:00, 688.87it/s][A
    batches:  86%|████████▌ | 3224/3750 [00:04<00:00, 690.55it/s][A
    batches:  88%|████████▊ | 3295/3750 [00:04<00:00, 694.81it/s][A
    batches:  90%|████████▉ | 3366/3750 [00:04<00:00, 698.20it/s][A
    batches:  92%|█████████▏| 3437/3750 [00:04<00:00, 699.06it/s][A
    batches:  94%|█████████▎| 3509/3750 [00:04<00:00, 703.08it/s][A
    batches:  95%|█████████▌| 3581/3750 [00:05<00:00, 705.46it/s][A
    batches:  97%|█████████▋| 3653/3750 [00:05<00:00, 706.98it/s][A
    batches:  99%|█████████▉| 3725/3750 [00:05<00:00, 709.10it/s][A
    epochs:  80%|████████  | 24/30 [02:49<00:31,  5.25s/it]      [A
    batches:   0%|          | 0/3750 [00:00<?, ?it/s][A
    batches:   2%|▏         | 60/3750 [00:00<00:06, 598.57it/s][A
    batches:   4%|▎         | 134/3750 [00:00<00:05, 680.92it/s][A
    batches:   6%|▌         | 209/3750 [00:00<00:04, 710.79it/s][A
    batches:   8%|▊         | 284/3750 [00:00<00:04, 724.68it/s][A
    batches:  10%|▉         | 359/3750 [00:00<00:04, 733.01it/s][A
    batches:  12%|█▏        | 435/3750 [00:00<00:04, 740.35it/s][A
    batches:  14%|█▎        | 511/3750 [00:00<00:04, 744.38it/s][A
    batches:  16%|█▌        | 586/3750 [00:00<00:04, 726.46it/s][A
    batches:  18%|█▊        | 659/3750 [00:00<00:04, 707.89it/s][A
    batches:  20%|█▉        | 735/3750 [00:01<00:04, 721.30it/s][A
    batches:  22%|██▏       | 811/3750 [00:01<00:04, 732.75it/s][A
    batches:  24%|██▎       | 887/3750 [00:01<00:03, 738.68it/s][A
    batches:  26%|██▌       | 963/3750 [00:01<00:03, 742.29it/s][A
    batches:  28%|██▊       | 1039/3750 [00:01<00:03, 745.94it/s][A
    batches:  30%|██▉       | 1115/3750 [00:01<00:03, 748.31it/s][A
    batches:  32%|███▏      | 1191/3750 [00:01<00:03, 751.12it/s][A
    batches:  34%|███▍      | 1267/3750 [00:01<00:03, 751.53it/s][A
    batches:  36%|███▌      | 1343/3750 [00:01<00:03, 750.09it/s][A
    batches:  38%|███▊      | 1419/3750 [00:01<00:03, 751.13it/s][A
    batches:  40%|███▉      | 1495/3750 [00:02<00:03, 751.16it/s][A
    batches:  42%|████▏     | 1571/3750 [00:02<00:02, 753.00it/s][A
    batches:  44%|████▍     | 1647/3750 [00:02<00:02, 753.17it/s][A
    batches:  46%|████▌     | 1723/3750 [00:02<00:02, 752.55it/s][A
    batches:  48%|████▊     | 1799/3750 [00:02<00:02, 752.92it/s][A
    batches:  50%|█████     | 1875/3750 [00:02<00:02, 746.52it/s][A
    batches:  52%|█████▏    | 1953/3750 [00:02<00:02, 754.39it/s][A
    batches:  54%|█████▍    | 2029/3750 [00:02<00:02, 754.52it/s][A
    batches:  56%|█████▌    | 2105/3750 [00:02<00:02, 753.12it/s][A
    batches:  58%|█████▊    | 2181/3750 [00:02<00:02, 749.32it/s][A
    batches:  60%|██████    | 2257/3750 [00:03<00:01, 750.38it/s][A
    batches:  62%|██████▏   | 2333/3750 [00:03<00:01, 752.36it/s][A
    batches:  64%|██████▍   | 2409/3750 [00:03<00:01, 750.19it/s][A
    batches:  66%|██████▋   | 2485/3750 [00:03<00:01, 751.64it/s][A
    batches:  68%|██████▊   | 2561/3750 [00:03<00:01, 739.47it/s][A
    batches:  70%|███████   | 2635/3750 [00:03<00:01, 721.63it/s][A
    batches:  72%|███████▏  | 2710/3750 [00:03<00:01, 729.02it/s][A
    batches:  74%|███████▍  | 2787/3750 [00:03<00:01, 740.44it/s][A
    batches:  76%|███████▋  | 2863/3750 [00:03<00:01, 744.45it/s][A
    batches:  78%|███████▊  | 2939/3750 [00:03<00:01, 747.26it/s][A
    batches:  80%|████████  | 3015/3750 [00:04<00:00, 750.11it/s][A
    batches:  82%|████████▏ | 3091/3750 [00:04<00:00, 751.62it/s][A
    batches:  84%|████████▍ | 3168/3750 [00:04<00:00, 754.38it/s][A
    batches:  87%|████████▋ | 3244/3750 [00:04<00:00, 755.35it/s][A
    batches:  89%|████████▊ | 3321/3750 [00:04<00:00, 757.09it/s][A
    batches:  91%|█████████ | 3397/3750 [00:04<00:00, 749.69it/s][A
    batches:  93%|█████████▎| 3474/3750 [00:04<00:00, 753.57it/s][A
    batches:  95%|█████████▍| 3550/3750 [00:04<00:00, 751.32it/s][A
    batches:  97%|█████████▋| 3626/3750 [00:04<00:00, 751.84it/s][A
    batches:  99%|█████████▊| 3702/3750 [00:04<00:00, 746.06it/s][A
    epochs:  83%|████████▎ | 25/30 [02:54<00:25,  5.19s/it]      [A
    batches:   0%|          | 0/3750 [00:00<?, ?it/s][A
    batches:   2%|▏         | 63/3750 [00:00<00:05, 619.43it/s][A
    batches:   4%|▎         | 138/3750 [00:00<00:05, 692.02it/s][A
    batches:   6%|▌         | 211/3750 [00:00<00:04, 708.55it/s][A
    batches:   8%|▊         | 285/3750 [00:00<00:04, 718.93it/s][A
    batches:  10%|▉         | 357/3750 [00:00<00:04, 718.57it/s][A
    batches:  11%|█▏        | 431/3750 [00:00<00:04, 723.39it/s][A
    batches:  13%|█▎        | 504/3750 [00:00<00:04, 724.02it/s][A
    batches:  15%|█▌        | 579/3750 [00:00<00:04, 730.45it/s][A
    batches:  17%|█▋        | 655/3750 [00:00<00:04, 738.83it/s][A
    batches:  19%|█▉        | 729/3750 [00:01<00:04, 737.60it/s][A
    batches:  21%|██▏       | 803/3750 [00:01<00:03, 737.59it/s][A
    batches:  23%|██▎       | 879/3750 [00:01<00:03, 743.39it/s][A
    batches:  25%|██▌       | 956/3750 [00:01<00:03, 749.59it/s][A
    batches:  28%|██▊       | 1032/3750 [00:01<00:03, 751.59it/s][A
    batches:  30%|██▉       | 1108/3750 [00:01<00:03, 752.77it/s][A
    batches:  32%|███▏      | 1184/3750 [00:01<00:03, 753.56it/s][A
    batches:  34%|███▎      | 1260/3750 [00:01<00:03, 753.17it/s][A
    batches:  36%|███▌      | 1336/3750 [00:01<00:03, 753.10it/s][A
    batches:  38%|███▊      | 1412/3750 [00:01<00:03, 752.39it/s][A
    batches:  40%|███▉      | 1488/3750 [00:02<00:02, 754.20it/s][A
    batches:  42%|████▏     | 1564/3750 [00:02<00:02, 750.25it/s][A
    batches:  44%|████▎     | 1640/3750 [00:02<00:02, 748.29it/s][A
    batches:  46%|████▌     | 1716/3750 [00:02<00:02, 750.92it/s][A
    batches:  48%|████▊     | 1793/3750 [00:02<00:02, 754.80it/s][A
    batches:  50%|████▉     | 1869/3750 [00:02<00:02, 753.93it/s][A
    batches:  52%|█████▏    | 1945/3750 [00:02<00:02, 753.95it/s][A
    batches:  54%|█████▍    | 2021/3750 [00:02<00:02, 753.15it/s][A
    batches:  56%|█████▌    | 2097/3750 [00:02<00:02, 754.64it/s][A
    batches:  58%|█████▊    | 2173/3750 [00:02<00:02, 755.38it/s][A
    batches:  60%|█████▉    | 2249/3750 [00:03<00:01, 754.98it/s][A
    batches:  62%|██████▏   | 2325/3750 [00:03<00:01, 754.64it/s][A
    batches:  64%|██████▍   | 2401/3750 [00:03<00:01, 736.20it/s][A
    batches:  66%|██████▌   | 2477/3750 [00:03<00:01, 741.16it/s][A
    batches:  68%|██████▊   | 2552/3750 [00:03<00:01, 735.74it/s][A
    batches:  70%|███████   | 2626/3750 [00:03<00:01, 701.74it/s][A
    batches:  72%|███████▏  | 2698/3750 [00:03<00:01, 706.91it/s][A
    batches:  74%|███████▍  | 2771/3750 [00:03<00:01, 713.32it/s][A
    batches:  76%|███████▌  | 2843/3750 [00:03<00:01, 698.68it/s][A
    batches:  78%|███████▊  | 2914/3750 [00:03<00:01, 695.78it/s][A
    batches:  80%|███████▉  | 2987/3750 [00:04<00:01, 704.17it/s][A
    batches:  82%|████████▏ | 3058/3750 [00:04<00:01, 665.95it/s][A
    batches:  84%|████████▎ | 3132/3750 [00:04<00:00, 686.32it/s][A
    batches:  86%|████████▌ | 3207/3750 [00:04<00:00, 702.03it/s][A
    batches:  87%|████████▋ | 3281/3750 [00:04<00:00, 712.80it/s][A
    batches:  89%|████████▉ | 3356/3750 [00:04<00:00, 721.06it/s][A
    batches:  91%|█████████▏| 3429/3750 [00:04<00:00, 689.56it/s][A
    batches:  93%|█████████▎| 3499/3750 [00:04<00:00, 673.28it/s][A
    batches:  95%|█████████▌| 3567/3750 [00:04<00:00, 656.33it/s][A
    batches:  97%|█████████▋| 3636/3750 [00:05<00:00, 662.53it/s][A
    batches:  99%|█████████▉| 3704/3750 [00:05<00:00, 665.04it/s][A
    epochs:  87%|████████▋ | 26/30 [02:59<00:20,  5.19s/it]      [A
    batches:   0%|          | 0/3750 [00:00<?, ?it/s][A
    batches:   2%|▏         | 61/3750 [00:00<00:06, 606.61it/s][A
    batches:   4%|▎         | 137/3750 [00:00<00:05, 694.36it/s][A
    batches:   6%|▌         | 211/3750 [00:00<00:04, 713.97it/s][A
    batches:   8%|▊         | 286/3750 [00:00<00:04, 724.98it/s][A
    batches:  10%|▉         | 359/3750 [00:00<00:04, 718.15it/s][A
    batches:  12%|█▏        | 435/3750 [00:00<00:04, 730.92it/s][A
    batches:  14%|█▎        | 509/3750 [00:00<00:04, 720.00it/s][A
    batches:  16%|█▌        | 582/3750 [00:00<00:04, 706.32it/s][A
    batches:  17%|█▋        | 654/3750 [00:00<00:04, 708.92it/s][A
    batches:  19%|█▉        | 726/3750 [00:01<00:04, 711.53it/s][A
    batches:  21%|██▏       | 799/3750 [00:01<00:04, 714.91it/s][A
    batches:  23%|██▎       | 871/3750 [00:01<00:04, 709.85it/s][A
    batches:  25%|██▌       | 944/3750 [00:01<00:03, 714.58it/s][A
    batches:  27%|██▋       | 1016/3750 [00:01<00:03, 713.82it/s][A
    batches:  29%|██▉       | 1088/3750 [00:01<00:03, 711.03it/s][A
    batches:  31%|███       | 1166/3750 [00:01<00:03, 729.29it/s][A
    batches:  33%|███▎      | 1244/3750 [00:01<00:03, 743.72it/s][A
    batches:  35%|███▌      | 1321/3750 [00:01<00:03, 751.47it/s][A
    batches:  37%|███▋      | 1397/3750 [00:01<00:03, 745.94it/s][A
    batches:  39%|███▉      | 1472/3750 [00:02<00:03, 741.73it/s][A
    batches:  41%|████▏     | 1549/3750 [00:02<00:02, 748.27it/s][A
    batches:  43%|████▎     | 1626/3750 [00:02<00:02, 754.22it/s][A
    batches:  45%|████▌     | 1702/3750 [00:02<00:02, 743.08it/s][A
    batches:  47%|████▋     | 1777/3750 [00:02<00:02, 738.55it/s][A
    batches:  49%|████▉     | 1851/3750 [00:02<00:02, 736.66it/s][A
    batches:  51%|█████▏    | 1925/3750 [00:02<00:02, 735.26it/s][A
    batches:  53%|█████▎    | 1999/3750 [00:02<00:02, 728.38it/s][A
    batches:  55%|█████▌    | 2072/3750 [00:02<00:02, 721.41it/s][A
    batches:  57%|█████▋    | 2145/3750 [00:02<00:02, 719.75it/s][A
    batches:  59%|█████▉    | 2217/3750 [00:03<00:02, 717.03it/s][A
    batches:  61%|██████    | 2289/3750 [00:03<00:02, 715.06it/s][A
    batches:  63%|██████▎   | 2361/3750 [00:03<00:01, 716.36it/s][A
    batches:  65%|██████▍   | 2434/3750 [00:03<00:01, 718.21it/s][A
    batches:  67%|██████▋   | 2507/3750 [00:03<00:01, 718.99it/s][A
    batches:  69%|██████▉   | 2579/3750 [00:03<00:01, 718.38it/s][A
    batches:  71%|███████   | 2651/3750 [00:03<00:01, 718.84it/s][A
    batches:  73%|███████▎  | 2723/3750 [00:03<00:01, 716.21it/s][A
    batches:  75%|███████▍  | 2795/3750 [00:03<00:01, 714.27it/s][A
    batches:  76%|███████▋  | 2867/3750 [00:03<00:01, 712.63it/s][A
    batches:  78%|███████▊  | 2939/3750 [00:04<00:01, 713.01it/s][A
    batches:  80%|████████  | 3012/3750 [00:04<00:01, 715.34it/s][A
    batches:  82%|████████▏ | 3087/3750 [00:04<00:00, 722.90it/s][A
    batches:  84%|████████▍ | 3160/3750 [00:04<00:00, 709.73it/s][A
    batches:  86%|████████▌ | 3232/3750 [00:04<00:00, 704.23it/s][A
    batches:  88%|████████▊ | 3303/3750 [00:04<00:00, 692.08it/s][A
    batches:  90%|████████▉ | 3373/3750 [00:04<00:00, 690.63it/s][A
    batches:  92%|█████████▏| 3443/3750 [00:04<00:00, 689.51it/s][A
    batches:  94%|█████████▎| 3512/3750 [00:04<00:00, 688.20it/s][A
    batches:  95%|█████████▌| 3581/3750 [00:04<00:00, 686.75it/s][A
    batches:  97%|█████████▋| 3650/3750 [00:05<00:00, 687.44it/s][A
    batches:  99%|█████████▉| 3720/3750 [00:05<00:00, 690.09it/s][A
    epochs:  90%|█████████ | 27/30 [03:04<00:15,  5.21s/it]      [A
    batches:   0%|          | 0/3750 [00:00<?, ?it/s][A
    batches:   2%|▏         | 61/3750 [00:00<00:06, 601.57it/s][A
    batches:   3%|▎         | 130/3750 [00:00<00:05, 650.18it/s][A
    batches:   5%|▌         | 202/3750 [00:00<00:05, 681.82it/s][A
    batches:   7%|▋         | 274/3750 [00:00<00:04, 696.80it/s][A
    batches:   9%|▉         | 347/3750 [00:00<00:04, 705.98it/s][A
    batches:  11%|█         | 418/3750 [00:00<00:04, 692.34it/s][A
    batches:  13%|█▎        | 490/3750 [00:00<00:04, 699.44it/s][A
    batches:  15%|█▌        | 567/3750 [00:00<00:04, 720.46it/s][A
    batches:  17%|█▋        | 644/3750 [00:00<00:04, 734.84it/s][A
    batches:  19%|█▉        | 719/3750 [00:01<00:04, 739.22it/s][A
    batches:  21%|██        | 795/3750 [00:01<00:03, 745.14it/s][A
    batches:  23%|██▎       | 872/3750 [00:01<00:03, 751.98it/s][A
    batches:  25%|██▌       | 949/3750 [00:01<00:03, 755.43it/s][A
    batches:  27%|██▋       | 1026/3750 [00:01<00:03, 757.81it/s][A
    batches:  29%|██▉       | 1102/3750 [00:01<00:03, 754.96it/s][A
    batches:  31%|███▏      | 1179/3750 [00:01<00:03, 757.44it/s][A
    batches:  33%|███▎      | 1255/3750 [00:01<00:03, 746.10it/s][A
    batches:  35%|███▌      | 1330/3750 [00:01<00:03, 740.35it/s][A
    batches:  37%|███▋      | 1405/3750 [00:01<00:03, 728.93it/s][A
    batches:  39%|███▉      | 1480/3750 [00:02<00:03, 732.81it/s][A
    batches:  41%|████▏     | 1555/3750 [00:02<00:02, 736.97it/s][A
    batches:  43%|████▎     | 1630/3750 [00:02<00:02, 740.40it/s][A
    batches:  45%|████▌     | 1705/3750 [00:02<00:02, 742.38it/s][A
    batches:  47%|████▋     | 1780/3750 [00:02<00:02, 743.53it/s][A
    batches:  49%|████▉     | 1855/3750 [00:02<00:02, 744.10it/s][A
    batches:  51%|█████▏    | 1930/3750 [00:02<00:02, 743.72it/s][A
    batches:  53%|█████▎    | 2005/3750 [00:02<00:02, 726.78it/s][A
    batches:  55%|█████▌    | 2079/3750 [00:02<00:02, 729.05it/s][A
    batches:  57%|█████▋    | 2152/3750 [00:02<00:02, 719.86it/s][A
    batches:  59%|█████▉    | 2225/3750 [00:03<00:02, 716.06it/s][A
    batches:  61%|██████▏   | 2299/3750 [00:03<00:02, 720.80it/s][A
    batches:  63%|██████▎   | 2372/3750 [00:03<00:01, 719.38it/s][A
    batches:  65%|██████▌   | 2444/3750 [00:03<00:01, 707.33it/s][A
    batches:  67%|██████▋   | 2515/3750 [00:03<00:01, 699.78it/s][A
    batches:  69%|██████▉   | 2588/3750 [00:03<00:01, 708.32it/s][A
    batches:  71%|███████   | 2664/3750 [00:03<00:01, 722.18it/s][A
    batches:  73%|███████▎  | 2737/3750 [00:03<00:01, 709.48it/s][A
    batches:  75%|███████▍  | 2809/3750 [00:03<00:01, 710.69it/s][A
    batches:  77%|███████▋  | 2886/3750 [00:03<00:01, 726.25it/s][A
    batches:  79%|███████▉  | 2959/3750 [00:04<00:01, 725.62it/s][A
    batches:  81%|████████  | 3035/3750 [00:04<00:00, 733.87it/s][A
    batches:  83%|████████▎ | 3109/3750 [00:04<00:00, 723.28it/s][A
    batches:  85%|████████▍ | 3182/3750 [00:04<00:00, 680.79it/s][A
    batches:  87%|████████▋ | 3254/3750 [00:04<00:00, 690.51it/s][A
    batches:  89%|████████▊ | 3326/3750 [00:04<00:00, 698.33it/s][A
    batches:  91%|█████████ | 3399/3750 [00:04<00:00, 706.92it/s][A
    batches:  93%|█████████▎| 3473/3750 [00:04<00:00, 715.89it/s][A
    batches:  95%|█████████▍| 3547/3750 [00:04<00:00, 720.94it/s][A
    batches:  97%|█████████▋| 3620/3750 [00:05<00:00, 703.63it/s][A
    batches:  98%|█████████▊| 3691/3750 [00:05<00:00, 679.67it/s][A
    epochs:  93%|█████████▎| 28/30 [03:09<00:10,  5.21s/it]      [A
    batches:   0%|          | 0/3750 [00:00<?, ?it/s][A
    batches:   1%|▏         | 55/3750 [00:00<00:06, 543.77it/s][A
    batches:   3%|▎         | 124/3750 [00:00<00:05, 629.33it/s][A
    batches:   5%|▌         | 193/3750 [00:00<00:05, 653.56it/s][A
    batches:   7%|▋         | 268/3750 [00:00<00:05, 689.60it/s][A
    batches:   9%|▉         | 342/3750 [00:00<00:04, 704.36it/s][A
    batches:  11%|█         | 416/3750 [00:00<00:04, 715.01it/s][A
    batches:  13%|█▎        | 491/3750 [00:00<00:04, 724.77it/s][A
    batches:  15%|█▌        | 568/3750 [00:00<00:04, 736.36it/s][A
    batches:  17%|█▋        | 642/3750 [00:00<00:04, 737.42it/s][A
    batches:  19%|█▉        | 716/3750 [00:01<00:04, 727.61it/s][A
    batches:  21%|██        | 791/3750 [00:01<00:04, 731.73it/s][A
    batches:  23%|██▎       | 865/3750 [00:01<00:03, 724.20it/s][A
    batches:  25%|██▌       | 938/3750 [00:01<00:03, 722.22it/s][A
    batches:  27%|██▋       | 1011/3750 [00:01<00:03, 722.35it/s][A
    batches:  29%|██▉       | 1084/3750 [00:01<00:03, 721.71it/s][A
    batches:  31%|███       | 1157/3750 [00:01<00:03, 712.83it/s][A
    batches:  33%|███▎      | 1230/3750 [00:01<00:03, 716.72it/s][A
    batches:  35%|███▍      | 1303/3750 [00:01<00:03, 720.21it/s][A
    batches:  37%|███▋      | 1377/3750 [00:01<00:03, 724.85it/s][A
    batches:  39%|███▊      | 1450/3750 [00:02<00:03, 718.93it/s][A
    batches:  41%|████      | 1522/3750 [00:02<00:03, 715.94it/s][A
    batches:  43%|████▎     | 1594/3750 [00:02<00:03, 714.78it/s][A
    batches:  44%|████▍     | 1666/3750 [00:02<00:02, 710.76it/s][A
    batches:  46%|████▋     | 1738/3750 [00:02<00:02, 710.48it/s][A
    batches:  48%|████▊     | 1810/3750 [00:02<00:02, 704.11it/s][A
    batches:  50%|█████     | 1887/3750 [00:02<00:02, 721.41it/s][A
    batches:  52%|█████▏    | 1965/3750 [00:02<00:02, 737.26it/s][A
    batches:  54%|█████▍    | 2043/3750 [00:02<00:02, 748.50it/s][A
    batches:  57%|█████▋    | 2120/3750 [00:02<00:02, 754.61it/s][A
    batches:  59%|█████▊    | 2198/3750 [00:03<00:02, 758.25it/s][A
    batches:  61%|██████    | 2276/3750 [00:03<00:01, 762.12it/s][A
    batches:  63%|██████▎   | 2353/3750 [00:03<00:01, 764.00it/s][A
    batches:  65%|██████▍   | 2430/3750 [00:03<00:01, 749.20it/s][A
    batches:  67%|██████▋   | 2506/3750 [00:03<00:01, 750.20it/s][A
    batches:  69%|██████▉   | 2582/3750 [00:03<00:01, 748.52it/s][A
    batches:  71%|███████   | 2657/3750 [00:03<00:01, 748.12it/s][A
    batches:  73%|███████▎  | 2732/3750 [00:03<00:01, 554.35it/s][A
    batches:  75%|███████▍  | 2807/3750 [00:03<00:01, 599.53it/s][A
    batches:  77%|███████▋  | 2884/3750 [00:04<00:01, 641.25it/s][A
    batches:  79%|███████▉  | 2957/3750 [00:04<00:01, 664.68it/s][A
    batches:  81%|████████  | 3028/3750 [00:04<00:01, 677.09it/s][A
    batches:  83%|████████▎ | 3100/3750 [00:04<00:00, 688.41it/s][A
    batches:  85%|████████▍ | 3173/3750 [00:04<00:00, 699.19it/s][A
    batches:  87%|████████▋ | 3245/3750 [00:04<00:00, 698.78it/s][A
    batches:  88%|████████▊ | 3317/3750 [00:04<00:00, 704.93it/s][A
    batches:  90%|█████████ | 3389/3750 [00:04<00:00, 709.30it/s][A
    batches:  92%|█████████▏| 3461/3750 [00:04<00:00, 711.40it/s][A
    batches:  94%|█████████▍| 3539/3750 [00:04<00:00, 729.60it/s][A
    batches:  96%|█████████▋| 3618/3750 [00:05<00:00, 744.33it/s][A
    batches:  99%|█████████▊| 3697/3750 [00:05<00:00, 755.25it/s][A
    epochs:  97%|█████████▋| 29/30 [03:15<00:05,  5.22s/it]      [A
    batches:   0%|          | 0/3750 [00:00<?, ?it/s][A
    batches:   2%|▏         | 59/3750 [00:00<00:06, 582.03it/s][A
    batches:   3%|▎         | 131/3750 [00:00<00:05, 657.28it/s][A
    batches:   5%|▌         | 204/3750 [00:00<00:05, 687.96it/s][A
    batches:   7%|▋         | 277/3750 [00:00<00:04, 703.92it/s][A
    batches:   9%|▉         | 350/3750 [00:00<00:04, 711.62it/s][A
    batches:  11%|█▏        | 423/3750 [00:00<00:04, 714.59it/s][A
    batches:  13%|█▎        | 495/3750 [00:00<00:04, 715.11it/s][A
    batches:  15%|█▌        | 568/3750 [00:00<00:04, 717.23it/s][A
    batches:  17%|█▋        | 641/3750 [00:00<00:04, 719.57it/s][A
    batches:  19%|█▉        | 713/3750 [00:01<00:04, 718.30it/s][A
    batches:  21%|██        | 786/3750 [00:01<00:04, 720.54it/s][A
    batches:  23%|██▎       | 859/3750 [00:01<00:04, 720.91it/s][A
    batches:  25%|██▍       | 932/3750 [00:01<00:03, 719.70it/s][A
    batches:  27%|██▋       | 1004/3750 [00:01<00:03, 708.45it/s][A
    batches:  29%|██▊       | 1076/3750 [00:01<00:03, 711.55it/s][A
    batches:  31%|███       | 1148/3750 [00:01<00:03, 710.89it/s][A
    batches:  33%|███▎      | 1220/3750 [00:01<00:03, 710.65it/s][A
    batches:  34%|███▍      | 1292/3750 [00:01<00:03, 691.01it/s][A
    batches:  36%|███▋      | 1368/3750 [00:01<00:03, 708.91it/s][A
    batches:  39%|███▊      | 1444/3750 [00:02<00:03, 721.36it/s][A
    batches:  41%|████      | 1520/3750 [00:02<00:03, 730.40it/s][A
    batches:  43%|████▎     | 1594/3750 [00:02<00:03, 708.22it/s][A
    batches:  44%|████▍     | 1666/3750 [00:02<00:02, 711.01it/s][A
    batches:  46%|████▋     | 1742/3750 [00:02<00:02, 722.85it/s][A
    batches:  48%|████▊     | 1817/3750 [00:02<00:02, 730.67it/s][A
    batches:  50%|█████     | 1893/3750 [00:02<00:02, 737.32it/s][A
    batches:  52%|█████▏    | 1967/3750 [00:02<00:02, 722.46it/s][A
    batches:  54%|█████▍    | 2040/3750 [00:02<00:02, 717.86it/s][A
    batches:  56%|█████▋    | 2112/3750 [00:02<00:02, 687.97it/s][A
    batches:  58%|█████▊    | 2186/3750 [00:03<00:02, 702.59it/s][A
    batches:  60%|██████    | 2263/3750 [00:03<00:02, 720.15it/s][A
    batches:  62%|██████▏   | 2336/3750 [00:03<00:01, 717.44it/s][A
    batches:  64%|██████▍   | 2408/3750 [00:03<00:01, 689.69it/s][A
    batches:  66%|██████▌   | 2478/3750 [00:03<00:01, 688.64it/s][A
    batches:  68%|██████▊   | 2548/3750 [00:03<00:01, 684.03it/s][A
    batches:  70%|██████▉   | 2617/3750 [00:03<00:01, 676.74it/s][A
    batches:  72%|███████▏  | 2685/3750 [00:03<00:01, 640.61it/s][A
    batches:  74%|███████▎  | 2759/3750 [00:03<00:01, 668.44it/s][A
    batches:  76%|███████▌  | 2836/3750 [00:04<00:01, 695.21it/s][A
    batches:  78%|███████▊  | 2912/3750 [00:04<00:01, 712.36it/s][A
    batches:  80%|███████▉  | 2989/3750 [00:04<00:01, 728.55it/s][A
    batches:  82%|████████▏ | 3066/3750 [00:04<00:00, 738.03it/s][A
    batches:  84%|████████▍ | 3142/3750 [00:04<00:00, 743.22it/s][A
    batches:  86%|████████▌ | 3217/3750 [00:04<00:00, 702.19it/s][A
    batches:  88%|████████▊ | 3288/3750 [00:04<00:00, 697.59it/s][A
    batches:  90%|████████▉ | 3359/3750 [00:04<00:00, 697.98it/s][A
    batches:  92%|█████████▏| 3433/3750 [00:04<00:00, 707.56it/s][A
    batches:  94%|█████████▎| 3507/3750 [00:04<00:00, 714.28it/s][A
    batches:  95%|█████████▌| 3579/3750 [00:05<00:00, 710.45it/s][A
    batches:  97%|█████████▋| 3653/3750 [00:05<00:00, 717.82it/s][A
    batches:  99%|█████████▉| 3727/3750 [00:05<00:00, 723.20it/s][A
    epochs: 100%|██████████| 30/30 [03:20<00:00,  6.68s/it]      [A


The model is now trained! There's no need to retrieve the weights from the
device as you would by calling `model.cpu()` with PyTorch. PopTorch has
managed that step for us. We can now save and evaluate the model.

#### Use the same IPU for training and inference
After the model has been attached to the IPU and compiled after the first call
to the PopTorch model, it can be detached from the device. This allows PopTorch
to use a single device for training and inference (described below), rather
than using 2 IPUs (one for training and one for inference) when the device
is not detached. When using an IPU-POD system, detaching from the device will
be necessary when using a non-reconfigurable partition.


```python
poptorch_model.detachFromDevice()

```

#### Save the trained model
We can simply use PyTorch's API to save a model in a file, with the original
instance of `ClassificationModel` and not the wrapped model.


```python
torch.save(model.state_dict(), "classifier.pth")

```

### Evaluate the model
Since we have detached our model from it's training from it's training device,
the device is now free again and we can use it for the evaluation stage,
instead of using the CPU. It is a good idea to use an IPU when evaluating your
model on a CPU is slow - be it because the test dataset is large and/or the model
is complex - since IPUs are blazing [fast](https://www.graphcore.ai/posts/new-graphcore-ipu-benchmarks).

The steps taken below to define the model for evaluation essentially allow it
to run in inference mode. Therefore, you can follow the same steps to use
the model to make predictions once it has been deployed.

For this, it is first essential to switch the model to evaluation mode. This
step is realised as usual.


```python
model = model.eval()
```

To evaluate the model on the IPU, we will use the `poptorch.inferenceModel`
class, which has a similar API to `poptorch.trainingModel` except that it
doesn't need an optimizer, allowing evaluation of the model without calculating
gradients.


```python
poptorch_model_inf = poptorch.inferenceModel(model, options=opts)
```

Then we can instantiate a new PopTorch dataloader object as before in order to
efficiently batch our test dataset.


```python
test_dataloader = poptorch.DataLoader(opts, test_dataset, batch_size=32, num_workers=10)

```

This short loop over the test dataset is effectively all that is needed to
run the model and generate some predictions. When running the model in
inference, we can stop here and use the predictions as needed.

For evaluation, we can use `scikit-learn`'s standard classification metrics to
understand how well our model is performing. This usually takes a list of labels
and a list of predictions as the input, both in the same order. Let's make both
lists, and run our model in inference mode.


```python
predictions, labels = [], []

for data, label in test_dataloader:
    predictions += poptorch_model_inf(data).data.max(dim=1).indices
    labels += label
```

    Graph compilation: 100%|██████████| 100/100 [00:17<00:00]


A simple and widely-used performance metric for classification models is the
accuracy score, which simply counts how many predictions were right. But this
metric alone isn't enough. For example, it doesn't tell us how the model
performs with regard to the different classes in our data. We will therefore
use another popular metric: a confusion matrix, which tells how much our
model confuses a class for another.


```python
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

print(f"Eval accuracy: {100 * accuracy_score(labels, predictions):.2f}%")

cm = confusion_matrix(labels, predictions)
cm_plot = ConfusionMatrixDisplay(cm, display_labels=classes).plot(xticks_rotation='vertical')
```

    Eval accuracy: 89.32%



    
![png](walkthrough_syntax_keras3_executed_files/walkthrough_syntax_keras3_executed_58_1.png)
    


>Eval accuracy: 89.32%
![png](static/from_0_to_1_33_1.png)
As you can see, although we've got an accuracy score of ~88%, the model's
performance across the different classes isn't equal. Trousers are very well
classified, with more than 96-97% accuracy whereas shirts are harder to
classify with less than 60% accuracy, and it seems they often get confused
with T-shirts, pullovers and coats. So, some work is still required here to
improve your model for all the classes!

We can save this visualisation of the confusion matrix.


```python
cm_plot.figure_.savefig("confusion_matrix.png")
```

# Doing more with `poptorch.Options`
This class encapsulates the options that PopTorch and PopART will use
alongside our model. Some concepts, such as "batch per iteration" are
specific to the functioning of the IPU, and within this class some
calculations are made to reduce risks of errors and make it easier for
PyTorch users to use IPUs.

The list of these options is available in the [documentation](https://docs.graphcore.ai/projects/poptorch-user-guide/en/latest/overview.html#options).
Let's introduce here 4 of these options to get an idea of what they cover.

### `deviceIterations`
Remember the training loop we have discussed previously. A device iteration
is one cycle of that loop, which runs entirely on the IPU (the device), and
which starts with a new batch of data. This option specifies the number of
batches that is prepared by the host (CPU) for the IPU. The higher this
number, the less the IPU has to interact with the CPU, for example to request
and wait for data, so that the IPU can loop faster. However, the user will
have to wait for the IPU to go over all the iterations before getting the
results back. The maximum is the total number of batches in your dataset, and
the default value is 1.

### `replicationFactor`
This is the number of replicas of a model. A replica is a copy of a same
model on multiple devices. We use replicas as an implementation of data
parallelism, where a same model is served with several batches of data at the
same time but on different devices, so that the gradients can be pooled. To
achieve the same behaviour in pure PyTorch, you'd wrap your model with `torch.
nn.DataParallel`, but with PopTorch, this is an option. Of course, each
replica requires one IPU. So, if the `replictionFactor` is two, two IPUs are
required.

### `randomSeed`
The IPU has a different, on-device pseudo-random number generator (PRNG).
This option sets the seed for the PRNG on the IPU. This is equivalent to
using `torch.seed`.

### `useIpuModel`
An IPU Model is a simulation, running on a CPU, of an actual IPU. This can be
helpful if you're working in an environment where no IPUs are available but
still need to make progress on your code. However, the IPU Model doesn't
fully support replicated graphs and its numerical results can be slightly
different from what you would get with an actual IPU. You can learn more
about the IPU Model and its limitations with our [documentation]
(https://docs.graphcore.ai/projects/poplar-user-guide/en/latest/poplar_programs.html?highlight=ipu%20model#programming-with-poplar).

## How to set the options
These options are callable, and chainable as they return the instance. One
can therefore do as follows:


```python
opts = poptorch.Options().deviceIterations(20).replicationFactor(2).randomSeed(123).useIpuModel(True)

```

# Going further

Other tutorials will be made available in the future to explore more advanced
features and use cases for PopTorch. Make sure you've subscribed to our
newsletter to stay up to date.

In the meantime, to learn more about the IPU and the lower level Poplar
libraries and graph programming framework, you can go through our Poplar
tutorials and read our Poplar SDK overview.
