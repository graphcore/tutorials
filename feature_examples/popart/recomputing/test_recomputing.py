# Copyright (c) 2019 Graphcore Ltd. All rights reserved.


import recomputing
import pytest
import argparse
import json


class TestRecomputingPopART(object):
    """Tests for recomputing PopART code example"""

    @pytest.mark.ipus(1)
    @pytest.mark.category1
    def test_manual_recomputing_use_less_memory(self):
        args = argparse.Namespace(
            test=True, export=None, report=False, recomputing='ON')
        session = recomputing.main(args)

        graph_report = json.loads(session.getGraphReport())
        recomputing_memory = sum(
            graph_report['memory']['byTile']['total'])

        args = argparse.Namespace(
            test=True, export=None, report=False, recomputing='OFF')
        session = recomputing.main(args)

        graph_report = json.loads(session.getGraphReport())
        no_recomputing_memory = sum(
            graph_report['memory']['byTile']['total'])

        print("\n")
        print("Memory use (recomputing) -->", recomputing_memory)
        print("Memory use (no recomputing) -->", no_recomputing_memory)
        assert (recomputing_memory < no_recomputing_memory)

    @pytest.mark.ipus(1)
    @pytest.mark.category1
    def test_auto_recomputing(self):
        args = argparse.Namespace(
            test=True, export=None, report=False, recomputing='AUTO')
        session = recomputing.main(args)

        graph_report = json.loads(session.getGraphReport())
        mem = sum(
            graph_report['memory']['byTile']['total'])
        print("Memory use (auto recomputing) -->", mem)
        assert mem > 0


if __name__ == '__main__':
    pytest.main(args=[__file__, '-s'])
