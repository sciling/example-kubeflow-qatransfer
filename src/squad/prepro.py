from kfp.components import InputPath
from kfp.components import OutputPath


def prepro_basic(
    dataset_path: InputPath(str),
    train_ratio,
    glove_vec_size,
    mode,
    tokenizer,
    url,
    port,
    prepro_squad_dir: OutputPath(str),
):
    import nltk

    from squad.prepro import prepro

    nltk.download("punkt")

    train_ratio = float(train_ratio)
    glove_vec_size = int(glove_vec_size)
    port = int(port)

    def main():
        args = get_args()
        prepro(args)

    def get_args():
        source_dir = dataset_path + "/data/squad"
        target_dir = prepro_squad_dir + "/squad"
        glove_dir = dataset_path + "/data/glove"
        from types import SimpleNamespace

        args = SimpleNamespace(
            source_dir=source_dir,
            target_dir=target_dir,
            debug=False,
            train_ratio=train_ratio,
            glove_corpus="6B",
            glove_dir=glove_dir,
            glove_vec_size=glove_vec_size,
            mode=mode,
            single_path="",
            tokenizer=tokenizer,
            url=url,
            port=port,
            split=False,
        )
        print(args)
        return args

    main()