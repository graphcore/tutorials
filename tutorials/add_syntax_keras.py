import os
from collections import namedtuple
from itertools import groupby

LineMeta = namedtuple("LineMeta", "mode content")

if __name__ == '__main__':
    source_file = os.path.join('pytorch', 'tut1_basics', 'walkthrough.py')
    output_file = os.path.join('pytorch', 'tut1_basics', 'walkthrough_syntax_keras.py')

    with open(source_file, 'r') as sf:
        if os.path.exists(output_file):
            os.remove(output_file)

        with open(output_file, 'w') as of:
            lines_meta = []
            for line in sf.readlines():
                if line.startswith("#"):
                    mode = "comment"
                    line = line[2:]
                elif line == "":
                    mode = "break"
                else:
                    mode = "code"
                lm = LineMeta(mode, line)
                lines_meta.append(lm)

            for label, group in groupby(lines_meta, key=lambda x: x.mode):
                group = list(group)
                if group:
                    mode = group[0].mode
                    if mode == "comment":
                        of.write('"""' + os.linesep)
                        for lm in group:
                            of.write(lm.content)
                        of.write('"""' + os.linesep)
                    else:
                        for lm in group:
                            of.write(lm.content)
