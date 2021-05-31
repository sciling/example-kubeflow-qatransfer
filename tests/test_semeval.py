import os
import pathlib
import sys
import tempfile
import unittest

from unittest import TestCase


sys.path.append("..")


DATA_DIR = "%s/../data/test/semeval_files" % pathlib.Path(__file__).parent.absolute()
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


def download_squad():
    import zipfile

    import requests

    from tqdm import tqdm

    r_squad = requests.get(
        "http://github.com/sciling/qatransfer/releases/download/v0.1/save.zip "
    )
    total_size_in_bytes = int(r_squad.headers.get("content-length", 0))
    progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
    with tempfile.TemporaryFile() as tf:
        for chunk in r_squad.iter_content(chunk_size=1024):
            progress_bar.update(len(chunk))
            tf.write(chunk)
        with zipfile.ZipFile(tf, "r") as f:
            f.extractall(WORK_DIR)
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

    def test_train(self):
        from src.semeval.semeval_train import semeval_train

        download_squad()
        load_path = "/save/out/squad/basic/00/save/basic-2000"
        shared_path = "/save/out/squad/basic/00/shared.json"
        run_id = "00"
        sent_size_th = "10"
        ques_size_th = "10"
        num_epochs = "1"
        num_steps = "1"
        eval_period = "1"
        save_period = "1"
        model_path = tempfile.mkdtemp()
        try:
            semeval_train(
                WORK_DIR,
                DATA_DIR,
                load_path,
                shared_path,
                run_id,
                sent_size_th,
                ques_size_th,
                num_epochs,
                num_steps,
                eval_period,
                save_period,
                model_path,
            )
        except SystemExit:
            print("Finished successfully!")
        # Check model directory has all files
        self.assertIn("out", os.listdir(model_path))
        self.assertIn("semeval", os.listdir(model_path + "/out"))

    def test_generate_files(self):
        from src.semeval.generate_semeval_test_files import generate_semeval_test_files

        test_path = tempfile.mkdtemp()
        try:
            generate_semeval_test_files(
                DATA_DIR, DATA_DIR, "2001", "2002", "1", "00", "10", test_path
            )
        except SystemExit:
            print("Finished successfully!")
        # Check model directory has all files
        self.assertIn("semeval", os.listdir(test_path))
        self.assertIn("store", os.listdir(test_path + "/semeval"))
        self.assertIn("test-gold", os.listdir(test_path + "/semeval/store"))

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
