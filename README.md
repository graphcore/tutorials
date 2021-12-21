# Graphcore Tutorials

This repository contains tutorials, feature examples and simple applications to help you learn how to use
Graphcore IPUs.

If you are interested in finding out more about Graphcore, including
getting preview access to IPUs to run these examples, please register
your interest [here](https://www.graphcore.ai/product_info).

Please note we are not currently accepting pull requests or issues on this
repository. If you are actively using this repository and want to report any issues, please raise a ticket through the Graphcore support portal: https://www.graphcore.ai/support.

The latest version of the documentation for the Poplar software stack, and other developer resources, is available at https://www.graphcore.ai/developer.

>  The code presented here requires using Poplar SDK 2.4. Please check other branches of this repository for code compatible with previous releases.

Please install and enable the Poplar SDK following the instructions in the Getting Started guide for your IPU system.

Unless otherwise specified by a LICENSE file in a subdirectory, the LICENSE referenced at the top level applies to the files in this repository.

## Repository contents

### Tutorials

The [tutorials/](tutorials) folder contains tutorials to help you get started using the Poplar SDK and Graphcore tools.

* [tutorials/poplar](tutorials/poplar) - A set of tutorials to introduce the Poplar graph programming framework and the PopLibs libraries.
* [tutorials/pytorch](tutorials/pytorch) - A set of tutorials to introduce the PyTorch framework support for the IPU.
* [tutorials/tensorflow1](tutorials/tensorflow1) - A set of tutorials to introduce the TensorFlow 1 framework support for the IPU.
* [tutorials/tensorflow2](tutorials/tensorflow2) - A set of tutorials to introduce the TensorFlow 2 framework support for the IPU.
* [tutorials/popvision](tutorials/popvision) - A set of tutorials to introduce PopVision, our suite of graphical application analysis tools.

A complete list of available tutorials can be found in the [tutorials/](tutorials) folder.

The README files for the tutorials are best viewed on GitHub.

### Feature examples

The [feature_examples/](feature_examples) folder contains small code examples showing you how to use various software features when developing for IPUs. See the READMEs in each folder for details.

### Simple application examples

The [simple_applications/](simple_applications) folder contains example applications written in different frameworks targeting the IPU. See the READMEs in each folder for details on how to use these applications.

### Kernel benchmarks

The [kernel_benchmarks/](kernel_benchmarks) folder contains code for benchmarking the performance of some selected types of neural network layers on the IPU, using TensorFlow or our PopART framework.

### Code used in tech notes, videos and blogs

The [tech_notes_code/](tech_notes_code), [videos_code/](videos_code) and [blogs_code/](blogs_code) folders contain code used in Graphcore tech notes, videos and blogs (respectively).

### Utilities

The [utils/](utils) folder contains utilities libraries and scripts.