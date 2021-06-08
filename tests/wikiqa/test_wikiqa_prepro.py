import os
import pathlib
import sys
import tempfile
import unittest

from unittest import TestCase


sys.path.append("..")


DATA_DIR = "%s/../../data/test/wikiqa_files" % pathlib.Path(__file__).parent.absolute()
WORK_DIR = tempfile.mkdtemp()


def download():
    import os
    import tempfile
    import zipfile

    import requests

    from tqdm import tqdm

    dataset_path = WORK_DIR
    # Download WikiQA
    r = requests.get(
        "https://download.microsoft.com/download/E/5/F/E5FCFCEE-7005-4814-853D-DAA7C66507E0/WikiQACorpus.zip",
        stream=True,
    )
    total_size_in_bytes = int(r.headers.get("content-length", 0))
    progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
    with tempfile.TemporaryFile() as tf:
        for chunk in r.iter_content(chunk_size=128):
            progress_bar.update(len(chunk))
            tf.write(chunk)
        with zipfile.ZipFile(tf, "r") as f:
            f.extractall(dataset_path)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")

    # Download GloVe
    GLOVE_DIR = dataset_path + "/glove"
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

    # Download Squad
    r_squad = requests.get(
        "https://github.com/sciling/qatransfer/releases/download/v0.1/save_class.zip"
    )
    total_size_in_bytes = int(r_squad.headers.get("content-length", 0))
    progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
    with tempfile.TemporaryFile() as tf:
        for chunk in r_squad.iter_content(chunk_size=1024):
            progress_bar.update(len(chunk))
            tf.write(chunk)
        with zipfile.ZipFile(tf, "r") as f:
            f.extractall(dataset_path)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")

    print(os.listdir(dataset_path))


class TestAll(TestCase):
    def test_prepro(self):
        from src.wikiqa.wikiqa_prepro import prepro_class

        download()
        # Directory of the model
        wikiqa_path = tempfile.mkdtemp()
        dataset_path = WORK_DIR
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
