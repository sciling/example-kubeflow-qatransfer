try:
    from kfp.components import InputPath
    from kfp.components import OutputPath
except ImportError:

    def InputPath(c):
        return c

    def OutputPath(c):
        return c


def semeval_train(
    dataset_path: InputPath(str),
    semeval_path: InputPath(str),
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
    model_path: OutputPath(str),
):
    import tensorflow as tf

    from basic.cli import main

    input_dir = semeval_path + "/semeval"
    output_dir = model_path + "/out/semeval"
    full_load_path = dataset_path + load_path
    full_shared_path = dataset_path + shared_path
    tf.app.run(
        main,
        argv=[
            "./basic/cli.py",
            "--data_dir",
            input_dir,
            "--out_base_dir",
            output_dir,
            "--load_path",
            full_load_path,
            "--shared_path",
            full_shared_path,
            "--load_trained_model",
            "--run_id",
            run_id,
            "--load_shared",
            "--nocluster",
            "--sent_size_th",
            sent_size_th,
            "--ques_size_th",
            ques_size_th,
            "--num_epochs",
            num_epochs,
            "--num_steps",
            num_steps,
            "--eval_period",
            eval_period,
            "--save_period",
            save_period,
            "--device",
            device,
            "--device_type",
            device_type,
            "--num_gpus",
            num_gpus,
        ],
    )
