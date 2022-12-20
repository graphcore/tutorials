# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

# THIS FILE IS AUTOGENERATED. Rerun SST after editing source file: walkthrough.py

%pip install - r requirements.txt

import os
import subprocess
import pva
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

mnist_path = "../../../simple_applications/pytorch/mnist/mnist_poptorch.py"
os.environ["POPLAR_ENGINE_OPTIONS"] = '{"autoReport.all":"true", "autoReport.directory":"mydirectory"}'
subprocess.run(["python3", mnist_path])

# Grab the most recently modified profile.pop file
working_dir = Path(".").glob("./mydirectory/training/*.pop")
report_files = [f for f in working_dir if f.is_file()]
report_files.sort(reverse=True, key=lambda a: a.stat().st_mtime)
report_path = str(report_files[0])

# Open the file

report = pva.openReport(report_path)

print("Number of compute sets: ", report.compilation.graph.numComputeSets)
print("Number of tiles on target: ", report.compilation.target.numTiles)
print("Version of Poplar used: ", report.poplarVersion.string)

sum(step.ipus[0].cycles for step in report.execution.steps)


class TestVisitor(pva.ProgramVisitor):
    def visitOnTileExecute(self, onTileExecute):
        if len(onTileExecute.computeset.vertices) > 1:
            print(onTileExecute.name)


v = TestVisitor()
for s in report.execution.steps:
    s.program.accept(v)

matplotlib.use("Agg")

plt.plot([tile.memory.total.includingGaps for tile in report.compilation.tiles])
plt.xlabel("Tiles")
plt.ylabel("Bytes")
plt.savefig("MemoryByTilePlot.png")

# Generated:2022-10-20T15:09 Source:walkthrough.py SST:0.0.9
