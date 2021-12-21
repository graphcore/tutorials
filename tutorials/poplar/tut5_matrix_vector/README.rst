Tutorial 5: matrix-vector multiplication
----------------------------------------

This tutorial builds up a more complex calculation on vertices: multiplying a
matrix by a vector. Do not hesitate to read through the `Poplar and PopLibs User
Guide <https://docs.graphcore.ai/projects/poplar-user-guide/en/latest/index.html>`_
to complement this tutorial.
Use ``tut5_matrix_vector/start_here`` as your working directory.

The file ``matrix-mul-codelets.cpp`` contains the outline for the vertex code
that will perform a dot product. Its input and output fields are already
defined:

.. code-block:: c++

  class DotProductVertex : public Vertex {
  public:
    Input<Vector<float>> a;
    Input<Vector<float>> b;
    Output<float> out;
  }

* Complete the ``compute`` function of ``DotProductVertex``.

The host code follows a similar pattern to the host code in the previous
tutorials. To demonstrate other ways of running Poplar code, this tutorial
uses the host CPU as the target. This can be useful for functional testing
when initially developing code. It is faster to compile and, because it
only models a single tile, there is no need to worry about mapping tensors
or compute sets to tiles. (Note that the CPU target cannot be used for
profiling.)

There are three tensors defined for the input matrix, input vector
and output vector:

.. code-block:: c++

  Tensor matrix = graph.addVariable(FLOAT, {numRows, numCols}, "matrix");
  Tensor inputVector = graph.addVariable(FLOAT, {numCols}, "inputVector");
  Tensor outputVector = graph.addVariable(FLOAT, {numRows}, "outputVector");

The function ``buildMultiplyProgram`` creates the graph and control program for
performing the multiplication. The control program executes a single compute set
called ``mulCS``. This compute set consists of a vertex for each output element
of the output vector (in other words, one vertex for each row of the input
matrix).

The next task in this tutorial is to write the host code to add the vertices to
the compute set.

* Create a loop that performs ``numRows`` iterations, each of which will add a
  vertex to the graph.

  * Use the ``addVertex`` function of the graph object to add a vertex of type
    ``DotProductVertex`` to the ``mulCS`` compute set.

  * Use the final argument of ``addVertex`` to connect the fields of the
    vertex to the relevant tensor slices for that row. Each vertex takes one
    row of the matrix (you can use the index operator on the ``matrix``
    tensor), and the entire ``in`` tensor, and outputs to a single element of
    the ``out`` tensor.

After adding this code, you can build and run the example. A makefile is provided
to compile the program. You can build it by running ``make``

As you can see from the host program code, you'll need to provide two arguments
to the execution command that specify the size of the matrix. For example,
running the program as shown below will multiply a 40x50 matrix by a vector of
size 50:

.. code-block:: bash

  $ ./tut5_cpu 40 50

The host code includes a check that the result is correct.

(Optional) Using the IPU
........................

This section describes how to modify the program to use the IPU hardware.

* Copy ``tut5.cpp`` to ``tut5_ipu_hardware.cpp`` and open it in an editor.

* Add these include lines:

.. code-block:: c++

  #include <poplar/DeviceManager.hpp>
  #include <algorithm>

* Add the following lines at the start of ``main``:

.. code-block:: c++

  // Create the DeviceManager which is used to discover devices
  auto manager = DeviceManager::createDeviceManager();

  // Attempt to attach to a single IPU:
  auto devices = manager.getDevices(poplar::TargetType::IPU, 1);
  std::cout << "Trying to attach to IPU\n";
  auto it = std::find_if(devices.begin(), devices.end(), [](Device &device) {
     return device.attach();
  });

  if (it == devices.end()) {
    std::cerr << "Error attaching to device\n";
    return -1;
  }

  auto device = std::move(*it);
  std::cout << "Attached to IPU " << device.getId() << std::endl;

  auto target = device.getTarget();

This gets a list of all devices consisting of a single IPU that are attached to
the host and tries to attach to each one in turn until successful.
This is a useful approach if there are multiple users on the host.
It is also possible to get a specific device using its device-manager ID with the
``getDevice`` function.

* Replace the following line which creates a CPU target:

.. code-block:: c++

  Graph graph(Target::createCPUTarget());

with this code:

.. code-block:: c++

  Graph graph(target);

* Add tile mapping of variables after their declaration:

.. code-block:: c++

  graph.setTileMapping(matrix, 0);
  graph.setTileMapping(inputVector, 0);
  graph.setTileMapping(outputVector, 0);

Also, add tile mapping for each vertex in function ``buildMultiplyProgram``:

.. code-block:: c++

  for (unsigned i = 0; i < numRows; ++i) {
      auto v = graph.addVertex(mulCS,              // Put the vertex in the
                                                   // 'mulCS' compute set.
                               "DotProductVertex", // Create a vertex of this
                                                   // type.
                               {{"a", matrix[i]},  // Connect input 'a' of the
                                                   // vertex to a row of the
                                                   // matrix.
                                {"b", in},         // Connect input 'b' of the
                                                   // vertex to whole
                                                   // input vector.
                                {"out", out[i]}}); // Connect the output 'out'
                                                   // of the vertex to a single
                                                   // element of the output
                                                   // vector.
      graph.setTileMapping(v, i);
    }
    // The returned program just executes the 'mulCS' compute set i.e. executes
    // every vertex calculation in parallel.
    return Execute(mulCS);
  }

* Replace the following line:

.. code-block:: c++

  engine.load(Device::createCPUDevice());

with:

.. code-block:: c++

  engine.load(device);

* Compile the program.

.. code-block:: bash

  $ g++ --std=c++11 tut5_ipu_hardware.cpp -lpoplar -o tut5_ipu

Before running this you need to make sure that you have set the environment
variables for the Graphcore drivers (see the Getting Started Guide for your IPU
system).

* Run the program to see the same results as running on CPU

.. code-block:: bash

  $ ./tut5_ipu_hardware

Copyright (c) 2018 Graphcore Ltd. All rights reserved.
