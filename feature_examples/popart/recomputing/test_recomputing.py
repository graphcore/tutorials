# Copyright (c) 2019 Graphcore Ltd. All rights reserved.


import recomputing
import pytest
import argparse
import pva


class TestRecomputingPopART(object):
    """Tests for recomputing PopART code example"""

    @pytest.mark.ipus(1)
    @pytest.mark.category1
    def test_manual_recomputing_use_less_memory(self):
        args = argparse.Namespace(
            test=True, export=None, recomputing='ON')
        session = recomputing.main(args)

        report = session.getReport()
        recomputing_memory = sum(
            [t.memory.total.excludingGaps for t in report.compilation.tiles])

        args = argparse.Namespace(
            test=True, export=None, recomputing='OFF')
        session = recomputing.main(args)

        report = session.getReport()
        no_recomputing_memory = sum(
            [t.memory.total.excludingGaps for t in report.compilation.tiles])

        print("\n")
        print("Memory use (recomputing) -->", recomputing_memory)
        print("Memory use (no recomputing) -->", no_recomputing_memory)
        assert (recomputing_memory < no_recomputing_memory)

    @pytest.mark.ipus(1)
    @pytest.mark.category1
    def test_auto_recomputing(self):
        args = argparse.Namespace(
            test=True, export=None, recomputing='AUTO')
        session = recomputing.main(args)

        report = session.getReport()
        mem = sum(
            [t.memory.total.excludingGaps for t in report.compilation.tiles])
        print("Memory use (auto recomputing) -->", mem)
        assert mem > 0


if __name__ == '__main__':
    pytest.main(args=[__file__, '-s'])
