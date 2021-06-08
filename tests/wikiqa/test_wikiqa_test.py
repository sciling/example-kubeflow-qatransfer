import pathlib
import sys
import tempfile
import unittest


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


class TestAll(unittest.TestCase):
    def test_test(self):
        import json

        from src.wikiqa.wikiqa_test import wikiqa_test

        download()
        shared_path = "/save/out/squad/basic-class/00/shared.json"
        run_id = "00"
        prepro_dir = DATA_DIR
        prev_model_dir = DATA_DIR
        sent_size_th = "10"
        ques_size_th = "10"
        num_epochs = "1"
        num_steps = "1"
        eval_period = "1"
        save_period = "1"
        mlpipeline_metrics_path = tempfile.NamedTemporaryFile()
        model_dir = tempfile.mkdtemp()
        try:
            from multiprocessing import Process

            args = (
                WORK_DIR,
                prepro_dir,
                prev_model_dir,
                shared_path,
                run_id,
                sent_size_th,
                ques_size_th,
                num_epochs,
                num_steps,
                eval_period,
                save_period,
                mlpipeline_metrics_path.name,
                model_dir,
            )
            p = Process(target=wikiqa_test, args=args)
            p.start()
            p.join()
        except SystemExit:
            print("Finished successfully!")

        with open(mlpipeline_metrics_path.name, "r") as f:
            metrics = json.load(f)
        self.assertIn("metrics", list(metrics.keys()))
        self.assertEqual(2, len(list(metrics["metrics"])))
        self.assertEqual("accuracy-score", metrics["metrics"][0]["name"])
        self.assertEqual("loss", metrics["metrics"][1]["name"])
        mlpipeline_metrics_path.close()


if __name__ == "__main__":
    unittest.main()
