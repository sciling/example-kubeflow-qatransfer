import kfp
import kfp.dsl as dsl

from kfp.components import func_to_container_op


# Define the pipeline
@dsl.pipeline(name="pipeline_model_squad", description="")
def qa_pipeline(
    prepro_train_ratio: float = 0.7,
    prepro_glove_vec_size: int = 100,
    prepro_mode: str = "full",
    prepro_tokenizer: str = "PTB",
    prepro_url: str = "vision-server2.corp.ai2",
    prepro_port: int = 8000,
    train_sent_size_th: str = "150",
    train_ques_size_th: str = "100",
    train_num_epochs: str = "12",
    train_num_steps: str = "55",
    train_eval_period: str = "50",
    train_save_period: str = "10",
):
    # Creating containers from python functions
    from test import test

    from download import download
    from generate_metrics import generate_metrics
    from prepro import prepro_basic
    from train import train

    download_op = func_to_container_op(
        download,
        base_image="tensorflow/tensorflow:latest-gpu-py3",
        packages_to_install=["tqdm"],
    )
    squad_preprocess_op = func_to_container_op(
        prepro_basic,
        base_image="sciling/tensorflow:0.12.0-py3",
        packages_to_install=[
            "https://github.com/sciling/qatransfer/archive/refs/heads/master.zip#egg=qatransfer",
            "psutil",
        ],
    )
    squad_span_pretrain_op = func_to_container_op(
        train,
        base_image="sciling/tensorflow:0.12.0-py3",
        packages_to_install=[
            "https://github.com/sciling/qatransfer/archive/refs/heads/master.zip#egg=qatransfer",
            "psutil",
        ],
    )
    squad_test_op = func_to_container_op(
        test,
        base_image="sciling/tensorflow:0.12.0-py3",
        packages_to_install=[
            "https://github.com/sciling/qatransfer/archive/refs/heads/master.zip#egg=qatransfer"
        ],
    )
    generate_metrics_op = func_to_container_op(
        generate_metrics,
        base_image="sciling/tensorflow:0.12.0-py3",
        packages_to_install=[
            "https://github.com/sciling/qatransfer/archive/refs/heads/master.zip#egg=qatransfer"
        ],
    )

    dataset_path = download_op()
    prepro_span = squad_preprocess_op(
        dataset_path.output,
        prepro_train_ratio,
        prepro_glove_vec_size,
        prepro_mode,
        prepro_tokenizer,
        prepro_url,
        prepro_port,
    )
    model = squad_span_pretrain_op(
        prepro_span.output,
        train_sent_size_th,
        train_ques_size_th,
        train_num_epochs,
        train_num_steps,
        train_eval_period,
        train_save_period,
    ).set_memory_request("4G")
    metrics = squad_test_op(
        prepro_span.output,
        model.output,
        train_sent_size_th,
        train_ques_size_th,
        train_num_epochs,
        train_num_steps,
        train_eval_period,
        train_save_period,
    )
    generate_metrics_op(metrics.outputs["metrics"])


if __name__ == "__main__":
    # Compile pipeline to generate compressed YAML definition of the pipeline.
    kfp.compiler.Compiler().compile(qa_pipeline, "{}.zip".format("squad_pipeline"))
