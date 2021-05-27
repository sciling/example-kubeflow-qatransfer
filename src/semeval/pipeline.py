import kfp
import kfp.components as comp
import kfp.dsl as dsl


# Define the pipeline
@dsl.pipeline(name="question-answering-pipeline", description="")
def qa_pipeline(
    squad_url: str = "http://github.com/sciling/qatransfer/releases/download/v0.1/save.zip",
    squad_load_path: str = "/save/out/squad/basic/00/save/basic-2000",
    squad_shared_path: str = "/save/out/squad/basic/00/shared.json",
    train_run_id: str = "00",
    train_sent_size_th: str = "150",
    train_ques_size_th: str = "100",
    train_num_epochs: str = "12",
    train_num_steps: str = "55",
    train_eval_period: str = "50",
    train_save_period: str = "10",
    test_start_step: int = 2001,
    test_end_step: int = 2002,
    test_eval_period: int = 1,
    test_threshold: float = 0.5,
    test_th: int = 10,
    test_reranking_th: int = 10,
    test_format: str = "trec",
    test_verbose: bool = False,
    test_ignore_noanswer: bool = False,
):
    from download import download
    from generate_semeval_test_files import generate_semeval_test_files
    from semeval_prepro import semeval_prepro
    from semeval_test import semeval_test
    from semeval_train import semeval_train

    download_op = comp.func_to_container_op(
        download,
        base_image="tensorflow/tensorflow:latest-gpu-py3",
        packages_to_install=["tqdm"],
    )
    semeval_prepro_op = comp.func_to_container_op(
        semeval_prepro,
        base_image="sciling/tensorflow:0.12.0-py3",
        packages_to_install=[
            "https://github.com/sciling/qatransfer/archive/refs/heads/master.zip#egg=qatransfer"
        ],
    )
    semeval_train_op = comp.func_to_container_op(
        semeval_train,
        base_image="sciling/tensorflow:0.12.0-py3",
        packages_to_install=[
            "https://github.com/sciling/qatransfer/archive/refs/heads/master.zip#egg=qatransfer"
        ],
    )
    semeval_generate_test_files_op = comp.func_to_container_op(
        generate_semeval_test_files,
        base_image="sciling/tensorflow:0.12.0-py3",
        packages_to_install=[
            "https://github.com/sciling/qatransfer/archive/refs/heads/master.zip#egg=qatransfer"
        ],
    )
    semeval_test_op = comp.func_to_container_op(
        semeval_test,
        base_image="sciling/tensorflow:0.12.0-py3",
        packages_to_install=[
            "https://github.com/sciling/qatransfer/archive/refs/heads/master.zip#egg=qatransfer"
        ],
    )

    # Download
    dataset_path = download_op(squad_url)

    # Preprocess semeval
    semeval_prepro = semeval_prepro_op(dataset_path.output)

    # Train semeval with pretrained model SQUAD
    semeval_model = semeval_train_op(
        dataset_path.output,
        semeval_prepro.output,
        load_path=squad_load_path,
        shared_path=squad_shared_path,
        run_id=train_run_id,
        sent_size_th=train_sent_size_th,
        ques_size_th=train_ques_size_th,
        num_epochs=train_num_epochs,
        num_steps=train_num_steps,
        eval_period=train_eval_period,
        save_period=train_save_period,
    )

    # Generate files for testing
    test_files = semeval_generate_test_files_op(
        semeval_prepro.output,
        semeval_model.output,
        test_start_step,
        test_end_step,
        test_eval_period,
        train_run_id,
        test_threshold,
    )

    # Test
    semeval_test_op(
        test_files.output,
        train_run_id,
        test_start_step,
        test_end_step,
        test_th,
        test_reranking_th,
        test_format,
        test_verbose,
        test_ignore_noanswer,
    )


if __name__ == "__main__":
    # Compile pipeline to generate compressed YAML definition of the pipeline.
    kfp.compiler.Compiler().compile(qa_pipeline, "{}.zip".format("qa_pipeline"))
