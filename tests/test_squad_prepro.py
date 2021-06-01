import os
import pathlib
import sys
import tempfile
import unittest


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


class TestAll(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
