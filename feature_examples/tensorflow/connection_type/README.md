# Graphcore

## Device Connection Type example

This code example demonstrates how to control if and when an IPU device is acquired using the `device_connection.type` configuration option.

|Mode          |Description                                                          |
|--------------|---------------------------------------------------------------------|
|ALWAYS        | indicates that the system will attach when configuring the device.  |
|ON_DEMAND     | will defer connection to when the IPU is needed.                    |
|PRE_COMPILE   | will never try to attach to a device and anything which is meant to be executed on the device will return all zeros. Used to pre-compile Poplar programs on machines without IPUs.      |
|NEVER         | will never try to attach to a device. Used when compiling offline.  |

These options can be used to change relative ordering of compilation versus IPU acquisition.
* If `ALWAYS` is selected (default) then the IPU device will always be acquired before compilation.
* If `ON_DEMAND` is selected then the IPU device will only be acquired once it is required which can be after compilation.
* If `PRE_COMPILE` is selected then the IPU device is never acquired. Requires the
  TF_POPLAR_FLAGS='--executable_cache_path=/path/to/cache' environment variable to be set to the directory
  where the compiled executables will be placed.
* If `NEVER` is selected then the IPU device is never acquired.


### File structure

* `connection_type.py` Minimal example.
* `README.md` This file.
* `requirements.txt` Required packages for the tests.
* `test_connection_type.py` pytest tests

### How to use this demo

1) Prepare the TensorFlow environment.

   Install the Poplar SDK following the instructions in the Getting Started guide for your IPU system.
   Make sure to run the enable.sh script for Poplar and activate a Python virtualenv with a TensorFlow 1
   or TensorFlow 2 wheel from the Poplar SDK installed (use the version appropriate to your operating system).

2) Run the script.

   ```
   python3 connection_type.py --connection_type {ALWAYS,ON_DEMAND,NEVER}
   ```

   or

   ```
   TF_POPLAR_FLAGS='--executable_cache_path=/path/to/cache' python3 connection_type.py --connection_type PRE_COMPILE
   ```

3) Run the tests.

   ```
   pip3 install -r requirements.txt
   python3 -m pytest
   ```

   The test runs for each mode and checks:
    * The resultant tensor is valid if returned (not expected for NEVER).
    * The regular stderr trace for evidence that compilation and device attachment occur in the expected order.


