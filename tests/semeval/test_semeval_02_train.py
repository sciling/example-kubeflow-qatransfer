import os
import pathlib
import sys
import tempfile
import unittest

from unittest import TestCase


sys.path.append("..")


DATA_DIR = "%s/../../data/test" % pathlib.Path(__file__).parent.absolute()
WORK_DIR = "/tmp/semeval-tests"


def download_squad():
    import zipfile

    import requests

    from tqdm import tqdm

    if not os.path.exists(os.path.join(WORK_DIR, "save/out/squad/basic/00/save")):
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
    else:
        print("SQUAD models already downloaded")


class TestAll(TestCase):
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
        model_path = WORK_DIR
        device = "/cpu:0"
        device_type = "gpu"
        num_gpus = "1"

        try:
            from multiprocessing import Process

            args = (
                WORK_DIR,
                WORK_DIR,
                load_path,
                shared_path,
                run_id,
                sent_size_th,
                ques_size_th,
                num_epochs,
                num_steps,
                eval_period,
                save_period,
                device,
                device_type,
                num_gpus,
                model_path,
            )
            p = Process(target=semeval_train, args=args)
            p.start()
            p.join()
        except SystemExit:
            print("Finished successfully!")
        # Check model directory has all files
        self.assertIn("out", os.listdir(model_path))
        self.assertIn("semeval", os.listdir(model_path + "/out"))


if __name__ == "__main__":
    unittest.main()
