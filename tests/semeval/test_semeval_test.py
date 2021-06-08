import pathlib
import sys
import tempfile
import unittest

from unittest import TestCase


sys.path.append("..")

DATA_DIR = "%s/../../data/test/semeval_files" % pathlib.Path(__file__).parent.absolute()
WORK_DIR = tempfile.mkdtemp()


class TestAll(TestCase):
    def test_test(self):
        import json

        from src.semeval.semeval_test import semeval_test

        mlpipeline_metrics_path = tempfile.NamedTemporaryFile()

        semeval_test(
            DATA_DIR + "/generated_files",
            "00",
            "2001",
            "2002",
            "10",
            "10",
            "trec",
            "False",
            "Flase",
            mlpipeline_metrics_path.name,
        )

        with open(mlpipeline_metrics_path.name, "r") as f:
            metrics = json.load(f)
        self.assertIn("metrics", list(metrics.keys()))
        self.assertEqual(3, len(list(metrics["metrics"])))
        self.assertEqual("MAP-test-00-002001", metrics["metrics"][0]["name"])
        self.assertEqual("MRR-test-00-002001", metrics["metrics"][1]["name"])
        self.assertEqual("AvgRec-test-00-002001", metrics["metrics"][2]["name"])
        mlpipeline_metrics_path.close()


if __name__ == "__main__":
    unittest.main()
