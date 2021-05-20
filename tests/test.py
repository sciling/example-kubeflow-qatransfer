import sys
import unittest

from unittest import TestCase


sys.path.append("..")
import json
import os
import pathlib
import tempfile

from src.semeval_prepro import step2
from src.semeval_train import step3
from src.generate_semeval_test_files import step4
from src.semeval_test import step5


DATA_DIR = f"{pathlib.Path(__file__).parent.absolute()}/../data/test/minidataset/"

class TestAll(TestCase):
    def test_prepro(self):
        # Directory of the model
        semeval_path = tempfile.mkdtemp()
        dataset_path = 'PATH_TO_DATASET'
        step2(dataset_path, semeval_path)

        # Check model directory has all files
        self.assertIn("00", os.listdir(semeval_path))

    def test_train(self):
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
        step5(test_path, run_ids, start_step, end_step, th, reranking_th, op_format, verbose, ignore_noanswer)


if __name__ == "__main__":
    unittest.main()