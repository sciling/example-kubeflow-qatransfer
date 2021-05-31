import os
import pathlib
import sys
import tempfile
import unittest

from unittest import TestCase


sys.path.append("..")


DATA_DIR = "%s/../data/test/squad_files" % pathlib.Path(__file__).parent.absolute()
WORK_DIR = tempfile.mkdtemp()


def download_dataset():
    import json
    import tempfile
    import zipfile

    import requests

    from tqdm import tqdm

    # Download GloVe
    print("Downloading glove")
    GLOVE_DIR = WORK_DIR + "/data/glove"
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

    print("Finished downloading glove")

    # Download SQuAD
    SQUAD_DIR = WORK_DIR + "/data/squad"
    os.makedirs(SQUAD_DIR, exist_ok=True)
    r_train = requests.get(
        "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json"
    )
    squad_train_json = json.loads(r_train.text)
    with open(SQUAD_DIR + "/train-v1.1.json", "w") as f:
        json.dump(squad_train_json, f)
    print(os.listdir(SQUAD_DIR))

    r_dev = requests.get(
        "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
    )
    squad_dev_json = json.loads(r_dev.text)
    with open(SQUAD_DIR + "/dev-v1.1.json", "w") as f:
        json.dump(squad_dev_json, f)
    print(os.listdir(SQUAD_DIR))


class TestAll(TestCase):
    def test_prepro(self):
        from src.squad.prepro import prepro_basic

        download_dataset()
        # Directory of the model
        squad_path = tempfile.mkdtemp()
        dataset_path = WORK_DIR
        train_ratio = 0.9
        glove_vec_size = 100
        mode = "full"
        tokenizer = "PTB"
        url = "vision-server2.corp.ai2"
        port = 8000
        prepro_basic(
            dataset_path,
            train_ratio,
            glove_vec_size,
            mode,
            tokenizer,
            url,
            port,
            squad_path,
        )

        # Check model directory has all files
        self.assertIn("data_dev.json", os.listdir(squad_path + "/squad"))
        self.assertIn("data_test.json", os.listdir(squad_path + "/squad"))
        self.assertIn("data_train.json", os.listdir(squad_path + "/squad"))
        self.assertIn("shared_dev.json", os.listdir(squad_path + "/squad"))
        self.assertIn("shared_test.json", os.listdir(squad_path + "/squad"))
        self.assertIn("shared_train.json", os.listdir(squad_path + "/squad"))

    def test_train(self):
        # Directory of the model
        import tensorflow as tf

        from src.squad.train import train

        squad_path = DATA_DIR
        sent_size_th = "10"
        ques_size_th = "10"
        num_epochs = "1"
        num_steps = "1"
        eval_period = "1"
        save_period = "1"
        learning_rate = "0.5"
        batch_size = "60"
        hidden_size = "100"
        var_decay = "0.999"
        model_path = tempfile.mkdtemp()
        try:
            import argparse as _argparse

            tf.app.flags._global_parser = _argparse.ArgumentParser()
            train(
                squad_path,
                sent_size_th,
                ques_size_th,
                num_epochs,
                num_steps,
                eval_period,
                save_period,
                learning_rate,
                batch_size,
                hidden_size,
                var_decay,
                model_path,
            )

        except SystemExit:
            print("Finished successfully!")
        # Check model directory has all files
        self.assertIn("out", os.listdir(model_path))
        self.assertIn("squad", os.listdir(model_path + "/out"))

    def test_test(self):
        import json

        import tensorflow as tf

        from src.squad.test import test

        prepro_dir = DATA_DIR
        prev_model_dir = DATA_DIR
        sent_size_th = "10"
        ques_size_th = "10"
        num_epochs = "1"
        num_steps = "1"
        eval_period = "1"
        save_period = "1"
        learning_rate = "0.5"
        batch_size = "60"
        hidden_size = "100"
        var_decay = "0.999"
        mlpipeline_metrics_path = tempfile.NamedTemporaryFile()
        model_dir = tempfile.mkdtemp()
        try:
            test(
                prepro_dir,
                prev_model_dir,
                sent_size_th,
                ques_size_th,
                num_epochs,
                num_steps,
                eval_period,
                save_period,
                learning_rate,
                batch_size,
                hidden_size,
                var_decay,
                mlpipeline_metrics_path.name,
                model_dir,
            )

            mlpipeline_metrics_path.close()
        except SystemExit:
            print("Finished successfully!")

        with open(mlpipeline_metrics_path.name, "r") as f:
            metrics = json.load(f)
        self.assertIn("metrics", list(metrics.keys()))
        self.assertEqual(2, len(list(metrics["metrics"])))
        self.assertEqual("accuracy-score", metrics["metrics"][0]["name"])
        self.assertEqual("f1-score", metrics["metrics"][1]["name"])


if __name__ == "__main__":
    unittest.main()
