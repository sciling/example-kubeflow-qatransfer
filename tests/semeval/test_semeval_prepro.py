import os
import pathlib
import sys
import tempfile
import unittest

from unittest import TestCase


sys.path.append("..")


DATA_DIR = "%s/../../data/test/semeval_files" % pathlib.Path(__file__).parent.absolute()
WORK_DIR = tempfile.mkdtemp()


def download_dataset():
    import zipfile

    import requests

    from tqdm import tqdm

    # Download SemEval
    SEMEVAL = WORK_DIR + "/semeval"
    os.makedirs(SEMEVAL, exist_ok=True)
    r_semeval = requests.get(
        "http://alt.qcri.org/semeval2016/task3/data/uploads/semeval2016-task3-cqa-ql-traindev-v3.2.zip"
    )
    total_size_in_bytes = int(r_semeval.headers.get("content-length", 0))
    progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
    with tempfile.TemporaryFile() as tf:
        for chunk in r_semeval.iter_content(chunk_size=1024):
            progress_bar.update(len(chunk))
            tf.write(chunk)
        with zipfile.ZipFile(tf, "r") as f:
            f.extractall(SEMEVAL)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")

    r_semeval_test = requests.get(
        "http://alt.qcri.org/semeval2016/task3/data/uploads/semeval2016_task3_test.zip"
    )
    total_size_in_bytes = int(r_semeval_test.headers.get("content-length", 0))
    progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
    with tempfile.TemporaryFile() as tf:
        for chunk in r_semeval_test.iter_content(chunk_size=1024):
            progress_bar.update(len(chunk))
            tf.write(chunk)
        with zipfile.ZipFile(tf, "r") as f:
            f.extractall(SEMEVAL)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")

    # Download GloVe
    GLOVE_DIR = WORK_DIR + "/glove"
    os.makedirs(GLOVE_DIR, exist_ok=True)
    r = requests.get("http://nlp.stanford.edu/data/glove.6B.zip", stream=True)
    total_size_in_bytes = int(r.headers.get("content-length", 0))
    progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
    with tempfile.TemporaryFile() as tf:
        for chunk in r.iter_content(chunk_size=1024):
            progress_bar.update(len(chunk))
            tf.write(chunk)
        with zipfile.ZipFile(tf, "r") as f:
            f.extractall(GLOVE_DIR)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")


class TestAll(TestCase):
    def test_prepro(self):
        from src.semeval.semeval_prepro import semeval_prepro

        download_dataset()
        # Directory of the model
        semeval_path = tempfile.mkdtemp()
        dataset_path = WORK_DIR
        semeval_prepro(dataset_path, semeval_path)

        # Check model directory has all files
        self.assertIn("data_dev.json", os.listdir(semeval_path + "/semeval"))
        self.assertIn("data_test.json", os.listdir(semeval_path + "/semeval"))
        self.assertIn("data_train.json", os.listdir(semeval_path + "/semeval"))


if __name__ == "__main__":
    unittest.main()
