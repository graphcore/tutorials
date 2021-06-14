#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

"""
A selection of examples for how to use the PopVision Analysis Library (libpva) with a Poplar
profile file (profile.pop).
"""
import pva
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

report = pva.openReport('profile.pop')

print("Number of compute sets: ", report.compilation.graph.numComputeSets)
print("Number of tiles on target: ", report.compilation.target.numTiles)
print("Version of Poplar used: ", report.poplarVersion.string)

print("Total number of cycles for all steps: ",
      sum(step.ipus[0].cycles for step in report.execution.steps))


class TestVisitor(pva.ProgramVisitor):
    def visitOnTileExecute(self, onTileExecute):
        if len(onTileExecute.computeset.vertices) > 1:
            print(onTileExecute.name)

print("The name of every onTileExecute program with more than one vertex:")

v = TestVisitor()
for s in report.execution.steps:
    s.program.accept(v)

plt.plot([tile.memory.total.includingGaps for tile in report.compilation.tiles])
plt.xlabel('Tiles')
plt.ylabel('Bytes')
plot_file = 'MemoryByTilePlot.png'
plt.savefig(plot_file)
print('Plot saved to {}.'.format(plot_file))
