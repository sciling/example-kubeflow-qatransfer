from kfp.components import InputPath
from kfp.components import OutputPath


def prepro_class(dataset_path: InputPath(str), wikiqa_path: OutputPath(str)):
    import nltk

    from wikiqa.prepro_class import prepro

    nltk.download("punkt")

    def get_args():
        from types import SimpleNamespace

        source_dir = dataset_path + "/WikiQACorpus"
        target_dir = wikiqa_path + "/wikiqa-class"
        glove_dir = dataset_path + "/glove"
        args = SimpleNamespace(
            source_dir=source_dir,
            target_dir=target_dir,
            debug=False,
            glove_corpus="6B",
            glove_dir=glove_dir,
            glove_vec_size="100",
            tokenizer="PTB",
        )
        return args

    args = get_args()
    prepro(args)
