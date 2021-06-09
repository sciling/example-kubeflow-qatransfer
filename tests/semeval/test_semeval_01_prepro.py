import os
import pathlib
import sys
import tempfile
import unittest

from unittest import TestCase


sys.path.append("..")


DATA_DIR = "%s/../../data/test" % pathlib.Path(__file__).parent.absolute()
WORK_DIR = "/tmp/semeval-tests"


class TestAll(TestCase):
    def test_prepro(self):
        from src.semeval.semeval_prepro import semeval_prepro

        # Directory of the model
        semeval_path = WORK_DIR
        os.makedirs(semeval_path, exist_ok=True)

        dataset_path = DATA_DIR
        semeval_prepro(dataset_path, semeval_path)

        # Check model directory has all files
        self.assertIn("data_dev.json", os.listdir(semeval_path + "/semeval"))
        self.assertIn("data_test.json", os.listdir(semeval_path + "/semeval"))
        self.assertIn("data_train.json", os.listdir(semeval_path + "/semeval"))
        print(semeval_path)


if __name__ == "__main__":
    unittest.main()
