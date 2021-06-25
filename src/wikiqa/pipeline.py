import kfp

from kfp.components import func_to_container_op


def qa_pipeline(
    squad_url: str = "https://github.com/sciling/qatransfer/releases/download/v0.1/save_class.zip",
    squad_load_path: str = "/save/out/squad/basic-class/00/save/basic-class-1",
    squad_shared_path: str = "/save/out/squad/basic-class/00/shared.json",
    run_id: str = "00",
    sent_size_th: str = "500",
    ques_size_th: str = "30",
    num_epochs: str = "12",
    num_steps: str = "5000",
    eval_period: str = "200",
    save_period: str = "200",
    start_step: int = 2001,
    end_step: int = 2201,
    device: str = "/cpu:0",
    device_type: str = "gpu",
    num_gpus: int = 1,
):
    from download import download
    from wikiqa_evaluate import wikiqa_evaluate
    from wikiqa_prepro import prepro_class
    from wikiqa_test import wikiqa_test
    from wikiqa_train import wikiqa_train

    download_op = func_to_container_op(
        download,
        base_image="tensorflow/tensorflow:latest-gpu-py3",
        packages_to_install=["tqdm"],
    )

    wikiqa_prepro_op = func_to_container_op(
        prepro_class,
        base_image="sciling/tensorflow:0.12.0-gpu-py3",
        packages_to_install=[
            "https://github.com/sciling/qatransfer/archive/refs/heads/master.zip#egg=qatransfer"
        ],
    )

    train_op = func_to_container_op(
        wikiqa_train,
        base_image="sciling/tensorflow:0.12.0-gpu-py3",
        packages_to_install=[
            "https://github.com/sciling/qatransfer/archive/refs/heads/master.zip#egg=qatransfer"
        ],
    )

    evaluate_op = func_to_container_op(
        wikiqa_evaluate,
        base_image="sciling/tensorflow:0.12.0-gpu-py3",
        packages_to_install=[
            "https://github.com/sciling/qatransfer/archive/refs/heads/master.zip#egg=qatransfer"
        ],
    )

    test_op = func_to_container_op(
        wikiqa_test,
        base_image="sciling/tensorflow:0.12.0-gpu-py3",
        packages_to_install=[
            "https://github.com/sciling/qatransfer/archive/refs/heads/master.zip#egg=qatransfer"
        ],
    )

    # Download
    dataset_path = download_op(squad_url)

    # Preprocess wikiqa
    wikiqa_prepro = wikiqa_prepro_op(dataset_path.output)

    # Train wikiqa with pretrained model SQUAD
    trained_model = train_op(
        dataset_path.output,
        wikiqa_prepro.output,
        load_path=squad_load_path,
        shared_path=squad_shared_path,
        run_id=run_id,
        sent_size_th=sent_size_th,
        ques_size_th=ques_size_th,
        num_epochs=num_epochs,
        num_steps=num_steps,
        eval_period=eval_period,
        save_period=save_period,
        device=device,
        device_type=device_type,
        num_gpus=num_gpus,
    )

    evaluate_op(
        wikiqa_prepro.output,
        trained_model.output,
        start_step,
        end_step,
        eval_period,
        run_id,
    )

    test_op(
        dataset_path.output,
        wikiqa_prepro.output,
        trained_model.output,
        shared_path=squad_shared_path,
        run_id=run_id,
        sent_size_th=sent_size_th,
        ques_size_th=ques_size_th,
        num_epochs=num_epochs,
        num_steps=num_steps,
        eval_period=eval_period,
        save_period=save_period,
        device=device,
        device_type=device_type,
        num_gpus=num_gpus,
    )


if __name__ == "__main__":
    # Compile pipeline to generate compressed YAML definition of the pipeline.
    kfp.compiler.Compiler().compile(qa_pipeline, "{}.zip".format("wikiqa_pipeline"))
