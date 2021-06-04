try:
    from kfp.components import OutputPath
except ImportError:

    def InputPath(c):
        return c

    def OutputPath(c):
        return c


def download(squad_url, dataset_path: OutputPath()):
    import os
    import tempfile
    import zipfile

    import requests

    from tqdm import tqdm

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
    r_squad = requests.get(squad_url)
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
