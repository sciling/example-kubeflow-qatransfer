import os
import pathlib
import sys
import unittest

from unittest import TestCase


sys.path.append("..")


DATA_DIR = "%s/../../data/test" % pathlib.Path(__file__).parent.absolute()
WORK_DIR = '/tmp/wikiqa-tests'


class TestAll(TestCase):
    def test_prepro(self):
        from src.wikiqa.wikiqa_prepro import prepro_class
        # Directory of the model
        # Directory of the model
        wikiqa_path = WORK_DIR
        os.makedirs(wikiqa_path, exist_ok=True)

        dataset_path = DATA_DIR
        prepro_class(dataset_path, wikiqa_path)

        # Check model directory has all files
        self.assertIn("data_dev.json", os.listdir(wikiqa_path + "/wikiqa-class"))
        self.assertIn("data_test.json", os.listdir(wikiqa_path + "/wikiqa-class"))
        self.assertIn("data_train.json", os.listdir(wikiqa_path + "/wikiqa-class"))
        self.assertIn("shared_dev.json", os.listdir(wikiqa_path + "/wikiqa-class"))
        self.assertIn("shared_test.json", os.listdir(wikiqa_path + "/wikiqa-class"))
        self.assertIn("shared_train.json", os.listdir(wikiqa_path + "/wikiqa-class"))
        self.assertIn("train-class.json", os.listdir(wikiqa_path + "/wikiqa-class"))
        self.assertIn("test-class.json", os.listdir(wikiqa_path + "/wikiqa-class"))
        self.assertIn("dev-class.json", os.listdir(wikiqa_path + "/wikiqa-class"))


if __name__ == "__main__":
    unittest.main()
