from kfp.components import OutputPath


def download(dataset_path: OutputPath(str)):
    import json
    import os
    import tempfile
    import zipfile

    import requests

    from tqdm import tqdm

    # Download GloVe
    print("Downloading glove")
    GLOVE_DIR = dataset_path + "/data/glove"
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
    SQUAD_DIR = dataset_path + "/data/squad"
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
