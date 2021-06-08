import pathlib
import sys
import tempfile
import os
import unittest

from unittest import TestCase


sys.path.append("..")

DATA_DIR = "%s/../../data/test" % pathlib.Path(__file__).parent.absolute()
WORK_DIR = '/tmp/semeval-tests'


class TestAll(TestCase):
    def test_test(self):
        import json

        from src.semeval.semeval_test import semeval_test

        mlpipeline_metrics_path = os.path.join(WORK_DIR, 'semeval/metric.json')

        semeval_test(
            WORK_DIR,
            "00",
            "2001",
            "2002",
            "10",
            "10",
            "trec",
            "False",
            "Flase",
            mlpipeline_metrics_path,
        )

        with open(mlpipeline_metrics_path, "r") as f:
            metrics = json.load(f)
        self.assertIn("metrics", list(metrics.keys()))
        self.assertEqual(3, len(list(metrics["metrics"])))
        self.assertEqual("MAP-test-00-002001", metrics["metrics"][0]["name"])
        self.assertEqual("MRR-test-00-002001", metrics["metrics"][1]["name"])
        self.assertEqual("AvgRec-test-00-002001", metrics["metrics"][2]["name"])


if __name__ == "__main__":
    unittest.main()
