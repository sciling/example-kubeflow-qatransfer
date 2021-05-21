import sys
import unittest

from unittest import TestCase


sys.path.append("..")
import json
import os
import pathlib
import tempfile

from src.generate_semeval_test_files import generate_semeval_test_files
from src.semeval_prepro import semeval_prepro
from src.semeval_test import semeval_test
from src.semeval_train import semeval_train


# DATA_DIR = f"{pathlib.Path(__file__).parent.absolute()}/../data/test/"


class TestAll(TestCase):
    def prepro_test(self):
        import zipfile

        import requests
        import tqdm

        # Download SemEval
        SEMEVAL = "/semeval"
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
        GLOVE_DIR = "/glove"
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

        # Directory of the model
        semeval_path = tempfile.mkdtemp()
        dataset_path = "."
        semeval_prepro(dataset_path, semeval_path)

        # Check model directory has all files
        self.assertIn(
            "shared_train_000_000.json", os.listdir(semeval_path + "/semeval")
        )
        self.assertGreater(len(list(os.listdir(semeval_path + "/semeval"))), 0)

    """def test_train(self):
        # Directory of the model
        dataset_path = 'PATH_TO_DATASET'
        semeval_path = 'PATH_TP_PREPRO_FILES'
        load_path = '/save/out/squad/basic/00/save/basic-2000'
        shared_path = '/save/out/squad/basic/00/shared.json'
        run_id = '00'
        sent_size_th = '150'
        ques_size_th = '100'
        num_epochs = '12'
        num_steps = '55'
        eval_period = '50'
        save_period = '10'
        model_path = tempfile.mkdtemp()
        step3(dataset_path, semeval_path, load_path, shared_path,
              run_id, sent_size_th, ques_size_th, num_epochs, num_steps, eval_period, save_period, model_path)

        # Check model directory has all files
        self.assertIn("model", os.listdir(model_path))

    def test_generate_files(self):
        semeval_path = 'PATH_TP_PREPRO_FILES'
        model_path = 'PATH_TP_PREPRO_FILES'
        run_ids = '00'
        start_step = 2001
        end_step = 2002
        eval_period = 1
        threshold = 0.5
        test_path = tempfile.mkdtemp()

        step4(semeval_path, model_path, start_step, end_step,
              eval_period, run_ids, threshold, test_path)

        # Check model directory has all files
        self.assertIn("gold_test", os.listdir(test_path))

    def test_test(self):
        test_path = 'PATH_TO_TEST'
        run_ids = '00'
        start_step = 2001
        end_step = 2002
        th = 10
        reranking_th = 10
        op_format = 'trec'
        verbose = False
        ignore_noanswer = False
        step5(test_path, run_ids, start_step, end_step, th, reranking_th, op_format, verbose, ignore_noanswer)"""


if __name__ == "__main__":
    unittest.main()
