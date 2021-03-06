try:
    from kfp.components import InputPath
    from kfp.components import OutputPath
except ImportError:

    def InputPath(c):
        return c

    def OutputPath(c):
        return c


def train(
    prepro_dir: InputPath(str),
    sent_size_th,
    ques_size_th,
    num_epochs,
    num_steps,
    eval_period,
    save_period,
    learning_rate,
    batch_size,
    hidden_size,
    var_decay,
    training_mode,
    device,
    device_type,
    num_gpus,
    model_dir: OutputPath(str),
):
    import tensorflow as tf

    from basic.cli import main

    model_name = "basic" if training_mode == "span" else "basic-class"
    data_dir = (
        prepro_dir + "/squad"
        if training_mode == "span"
        else prepro_dir + "/squad-class"
    )
    output_dir = model_dir + "/out/squad"
    argv = [
        "./basic/cli.py",
        "--model_name",
        model_name,
        "--data_dir",
        data_dir,
        "--out_base_dir",
        output_dir,
        "--noload",
        "--dev_name",
        "dev",
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
        "--init_lr",
        learning_rate,
        "--batch_size",
        batch_size,
        "--hidden_size",
        hidden_size,
        "--var_decay",
        var_decay,
        "--device",
        device,
        "--device_type",
        device_type,
        "--num_gpus",
        num_gpus,
    ]
    tf.app.run(main, argv)
