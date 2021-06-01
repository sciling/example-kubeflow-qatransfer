import pathlib
import sys
import tempfile
import unittest

from unittest import TestCase


sys.path.append("..")

DATA_DIR = "%s/../data/test/semeval_files" % pathlib.Path(__file__).parent.absolute()
WORK_DIR = tempfile.mkdtemp()


class TestAll(TestCase):
    def test_test(self):
        import sys

        from io import StringIO

        from src.semeval.semeval_test import semeval_test

        try:
            capturedOutput = StringIO()
            sys.stdout = capturedOutput
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
            )
        finally:
            sys.stdout = sys.__stdout__
        print(capturedOutput.getvalue())
        self.assertIn("MAP", capturedOutput.getvalue())
        self.assertIn("MRR", capturedOutput.getvalue())
        self.assertIn("AvgRec", capturedOutput.getvalue())


if __name__ == "__main__":
    unittest.main()
