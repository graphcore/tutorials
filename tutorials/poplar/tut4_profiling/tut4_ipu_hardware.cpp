// Copyright (c) 2018 Graphcore Ltd. All rights reserved.

/* This file contains the completed version of Poplar tutorial 4,
  which uses the IPU Hardware.
  See the Poplar user guide for details.
*/

#include <iostream>
#include <poplar/DeviceManager.hpp>
#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poplin/MatMul.hpp>
#include <poplin/codelets.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>

using namespace poplar;
using namespace poplar::program;

int main() {
  // Create the DeviceManager which is used to discover devices
  DeviceManager manager = DeviceManager::createDeviceManager();

  // Attempt to attach to a single IPU:
  Device device;
  bool success = false;
  // Loop over all single IPU devices on the host
  // Break the loop when an IPU is successfully acquired
  for (auto &hwDevice : manager.getDevices(poplar::TargetType::IPU, 1)) {
    device = std::move(hwDevice);
    std::cerr << "Trying to attach to IPU " << device.getId() << std::endl;
    if ((success = device.attach())) {
      std::cerr << "Attached to IPU " << device.getId() << std::endl;
      break;
    }
  }
  if (!success) {
    std::cerr << "Error attaching to device" << std::endl;
    return -1;
  }

  Target target = device.getTarget();

  // Create the Graph object
  Graph graph(target);
  popops::addCodelets(graph);
  poplin::addCodelets(graph);

  // Add variables to the graph
  Tensor m1 = graph.addVariable(FLOAT, {900, 600}, "m1");
  Tensor m2 = graph.addVariable(FLOAT, {600, 300}, "m2");
  Tensor m3 = graph.addVariable(FLOAT, {300, 200}, "m3");
  poputil::mapTensorLinearly(graph, m1);
  poputil::mapTensorLinearly(graph, m2);
  poputil::mapTensorLinearly(graph, m3);

  // Create a control program that is a sequence of steps
  Sequence prog;

  Tensor m4 = poplin::matMul(graph, m1, m2, prog, "m4");
  Tensor m5 = poplin::matMul(graph, m4, m3, prog, "m5");

  // Create the engine. We instruct the engine to perform instrumentation - this
  // adds cycle counters to the compiled program to enable the execution profile
  // to be retrieved after the program is run.
  Engine engine(graph, prog, {{"debug.instrument", "true"}});
  engine.load(device);

  // Run the control program
  std::cout << "Running program\n";
  engine.run(0);
  std::cout << "Program complete\n";

  engine.printProfileSummary(std::cout, {{"showExecutionSteps", "true"}});

  return 0;
}
