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
    def test_generate_files(self):
        from src.semeval.generate_semeval_test_files import generate_semeval_test_files

        test_path = WORK_DIR
        try:
            generate_semeval_test_files(
                WORK_DIR, WORK_DIR, "2001", "2002", "1", "00", "10", test_path
            )
        except SystemExit:
            print("Finished successfully!")
        # Check model directory has all files
        self.assertIn("semeval", os.listdir(test_path))
        self.assertIn("store", os.listdir(test_path + "/semeval"))
        self.assertIn("test-gold", os.listdir(test_path + "/semeval/store"))


if __name__ == "__main__":
    unittest.main()
