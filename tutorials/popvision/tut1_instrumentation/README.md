Tutorial 1: Instrumenting applications
--------------------------------------

For this tutorial we are going to use a PopART MNIST example and add instrumentation that can be viewed using the PopVision System Analyser. You can download the PopVision System Analyser from the Downloads Portal: <https://downloads.graphcore.ai/>

The PopART MNIST example is in the `start_here` directory (this is a subset of the code from the [simple_applications/popart/mnist](../../../simple_applications/popart/mnist) directory). Enter the `start_here` directory and follow the instructions in the README.md to install the required modules and download the data.

Once you have followed the instructions and been able to run the MNIST example we will re-run the MNIST example with instrumentation enabled. Note: A new report file will be created each time you run this command so you will need to close and open the new report each time.

    PVTI_OPTIONS='{"enable":"true"}' python3 popart_mnist.py


When this has completed you will find a pvti file in the working directory. e.g. "Tue_Nov_24_11:59:17_2020_GMT_4532.pvti". Note: You can specify an output directory for the pvti files to be written to

    PVTI_OPTIONS='{"enable":"true", "directory": "tommyFlowers"}' python3 popart_mnist.py

Open the PopVision System Analyser and then select "Open a report" and select the pvti file generated. You may need to copy the pvti file to your local machine.

You should then see the following profile information.

![PopVision System Analyser screenshot of mnist](./screenshots/mnist.png)

We are now going to modify the MNIST example to add instrumentation to clearly show the epochs. (You can find the completed tutorial in the `complete` directory)

Firstly, we need to import the libpvti library.

Add the import statement at the top of `popart_mnist.py`

    import libpvti as pvti

Next we will need to create a trace channel. Add the `mnistPvtiChannel` as a global object.

    mnistPvtiChannel = pvti.createTraceChannel("Mnist Application")

We are going to use the Python `with` keyword with a Python context manager to instrument the epoch loop. Note you will need to indent the contents of the loop

    print("Running training loop.")
    for i in range(opts.epochs):
      with pvti.Tracepoint(mnistPvtiChannel, f"Epoch:{i}"):
        ...

We leave it as an exercise for the reader to add instrumentation of the training & evaluation phases. When added you will see the following profile in the PopVision System Analyser. Note: You can nest the Tracepoint statements.

![PopVision System Analyser screenshot of instrumented mnist](./screenshots/mnist_instrumented.png)

Next, we are going to add instrumentation so the PopVision System Analyser can graph the loss reported by PopART. (This is a Poplar SDK 2.1 feature.)

We have added the libpvti import in the previous section, so we need first to create a pvti Graph object and then create series in the graph.

To create the graph we call the `pvti.Graph` constructor passing the name of the graph:

    loss_graph = pvti.Graph("Loss", "")

Then create the series to which we will add the data:

    training_loss_series = loss_graph.addSeries("Training Loss")
    validation_loss_series = loss_graph.addSeries("Validation Loss")

Finally after each call to the PopART `session.run` method we will record the training and validation loss. We take the loss from the anchors (which is an array) and compute the mean value:

    training.session.run(stepio, 'Epoch ' + str(i) + ' training step' + str(step))

    # Record the training loss
    training_loss_series.add(np.mean(training.anchors[loss]).item())

    ...

    validation.session.run(stepio, 'Epoch ' + str(i) + ' evaluation step ' + str(step))

    # Record the validation loss
    validation_loss_series.add(np.mean(validation.anchors[loss]).item())


When we view the resulting pvti report in the System Analyser (you may need to scroll to the bottom of the page) it will show the loss graph looking something like this:

![PopVision System Analyser screenshot of instrumented mnist loss](./screenshots/mnist_instrumented_loss.png)

(Note: The option to `merge all charts` has been enabled to combine all threads into a single row, to make it easier to align the flame graph with the line graph)

We leave it as an exercise for the reader to add additional instrumentation. The completed example also
records the graph for accuracy in the same way as loss and CPU load using the psutil library.

![PopVision System Analyser screenshot of instrumented mnist loss, accuracy & cpuload](./screenshots/mnist_instrumented_loss_accuracy_cpuload.png)


This is a very simple use case for adding instrumentation. The PopVision trace instrumentation library (libpvti) provides other functions, classes & methods to instrument your Python and C++ code. For more information please see the <a href=https://docs.graphcore.ai/projects/graphcore-popvision-user-guide/en/latest/system/system.html#using-the-libpvti-api>PVTI library section in the PopVision User Guide</a>.

Copyright (c) 2020 Graphcore Ltd. All rights reserved.
