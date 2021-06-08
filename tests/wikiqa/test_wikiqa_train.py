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
    def test_train(self):
        from src.wikiqa.wikiqa_train import wikiqa_train

        download()
        load_path = "/save/out/squad/basic-class/00/save/basic-class-1"
        shared_path = "/save/out/squad/basic-class/00/shared.json"
        run_id = "00"
        sent_size_th = "10"
        ques_size_th = "10"
        num_epochs = "1"
        num_steps = "1"
        eval_period = "1"
        save_period = "1"
        model_path = tempfile.mkdtemp()

        try:
            from multiprocessing import Process

            args = (
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
            p = Process(target=wikiqa_train, args=args)
            p.start()
            p.join()
        except SystemExit:
            print("Finished successfully!")

        # Check model directory has all files
        self.assertIn("out", os.listdir(model_path))
        self.assertIn("wikiqa", os.listdir(model_path + "/out"))


if __name__ == "__main__":
    unittest.main()
