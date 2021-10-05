# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
"""
This tool will automatically attach Keras-docs style syntax to a regular script, as long as it's formatted with some
rules:
- an empty newline split's cells
- markdowns will be made from regular comments
"""

import argparse
import os
from collections import namedtuple
from itertools import groupby

LineMeta = namedtuple("LineMeta", "mode content")
COMMENT = "comment"
BREAK = "break"
CODE = "code"
SEPARATOR = '"""'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', dest='source',
                        help='Source Python script without any style syntax')
    parser.add_argument('--target', dest='target',
                        help='Target Python script path which will receive style annotations')
    args = parser.parse_args()

    with open(args.source, 'r') as sf:
        if os.path.exists(args.target):
            os.remove(args.target)

        with open(args.target, 'w') as of:
            lines_meta = []
            for line in sf.readlines():
                if line.startswith("#"):
                    mode = COMMENT
                    line = line[2:]
                elif line.strip() == "":
                    mode = BREAK
                else:
                    mode = CODE
                lm = LineMeta(mode, line)
                lines_meta.append(lm)

            for label, group in groupby(lines_meta, key=lambda x: x.mode):
                group = list(group)
                if group:
                    mode = group[0].mode
                    if mode == COMMENT:
                        of.write(SEPARATOR + os.linesep)
                        for lm in group:
                            of.write(lm.content)
                        of.write(SEPARATOR + os.linesep)
                    elif mode == BREAK:
                        if len(group) == 1:
                            continue
                    else:
                        for lm in group:
                            of.write(lm.content)
