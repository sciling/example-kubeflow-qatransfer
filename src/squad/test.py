try:
    from kfp.components import InputPath
    from kfp.components import OutputPath
except ImportError:

    def InputPath(c):
        return c

    def OutputPath(c):
        return c


metrics = "Metrics"


def test(
    prepro_dir: InputPath(str),
    prev_model_dir: InputPath(str),
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
    mlpipeline_metrics_path: OutputPath(metrics),
    model_dir: OutputPath(str),
):

    import os
    import shutil

    import tensorflow as tf

    src = prev_model_dir + "/out/squad"
    dst = model_dir + "/out/squad"
    shutil.copytree(src, dst)

    model_name = "basic" if training_mode == "span" else "basic-class"
    data_dir = (
        prepro_dir + "/squad"
        if training_mode == "span"
        else prepro_dir + "/squad-class"
    )
    output_dir = model_dir + "/out/squad"

    flags = tf.app.flags

    # Names and directories
    flags.DEFINE_string("model_name", model_name, "Model name [basic | basic-class]")
    flags.DEFINE_string("data_dir", data_dir, "Data dir [data/squad]")
    flags.DEFINE_string("run_id", "0", "Run ID [0]")
    flags.DEFINE_string("out_base_dir", output_dir, "out base dir [out]")
    flags.DEFINE_string("forward_name", "single", "Forward name [single]")
    flags.DEFINE_string("answer_path", "", "Answer path []")
    flags.DEFINE_string("eval_path", "", "Eval path []")
    flags.DEFINE_string("load_path", "", "Load path []")
    flags.DEFINE_string("shared_path", "", "Shared path []")

    # Device placement
    flags.DEFINE_string(
        "device", device, "default device for summing gradients. [/cpu:0]"
    )
    flags.DEFINE_string(
        "device_type",
        device_type,
        "device for computing gradients (parallelization). cpu | gpu [gpu]",
    )
    flags.DEFINE_integer(
        "num_gpus", int(num_gpus), "num of gpus or cpus for computing gradients [1]"
    )

    # Essential training and test options
    flags.DEFINE_string("mode", "test", "train | test | forward [test]")
    flags.DEFINE_boolean("load", True, "load saved data? [True]")
    flags.DEFINE_bool("single", False, "supervise only the answer sentence? [False]")
    flags.DEFINE_boolean("debug", False, "Debugging mode? [False]")
    flags.DEFINE_bool(
        "load_ema", True, "load exponential average of variables when testing?  [True]"
    )
    flags.DEFINE_bool("eval", True, "eval? [True]")
    flags.DEFINE_bool("train_only_output", False, "Train only output module?")
    flags.DEFINE_bool("load_trained_model", False, "Load SQUAD trained model")
    flags.DEFINE_bool("freeze_phrase_layer", False, "Freeze phrase layer")
    flags.DEFINE_bool("freeze_att_layer", False, "Freeze att layer")
    flags.DEFINE_bool(
        "freeze_span_modelling_layer", False, "Freeze modelling layer for span"
    )

    flags.DEFINE_bool("using_shared", False, "using pre-created shared.json")
    flags.DEFINE_bool("load_shared", False, "load shared.json for each batch")
    flags.DEFINE_string("dev_name", "test", "using dev or test?")
    flags.DEFINE_string("test_name", "dev", "using test or dev?")

    # Training / test parameters
    flags.DEFINE_integer("batch_size", int(batch_size), "Batch size [60]")
    flags.DEFINE_integer("val_num_batches", 100, "validation num batches [100]")
    flags.DEFINE_integer("test_num_batches", 0, "test num batches [0]")
    flags.DEFINE_integer(
        "num_epochs", int(num_epochs), "Total number of epochs for training [12]"
    )
    flags.DEFINE_integer("num_steps", int(num_steps), "Number of steps [20000]")
    flags.DEFINE_integer("load_step", 0, "load step [0]")
    flags.DEFINE_float("init_lr", float(learning_rate), "Initial learning rate [0.5]")
    flags.DEFINE_float(
        "input_keep_prob", 0.8, "Input keep prob for the dropout of LSTM weights [0.8]"
    )
    flags.DEFINE_float(
        "keep_prob", 0.8, "Keep prob for the dropout of Char-CNN weights [0.8]"
    )
    flags.DEFINE_float("wd", 0.0, "L2 weight decay for regularization [0.0]")
    flags.DEFINE_integer("hidden_size", int(hidden_size), "Hidden size [100]")
    flags.DEFINE_integer("char_out_size", 100, "char-level word embedding size [100]")
    flags.DEFINE_integer("char_emb_size", 8, "Char emb size [8]")
    flags.DEFINE_string(
        "out_channel_dims",
        "100",
        "Out channel dims of Char-CNN, separated by commas [100]",
    )
    flags.DEFINE_string(
        "filter_heights", "5", "Filter heights of Char-CNN, separated by commas [5]"
    )
    flags.DEFINE_bool("finetune", False, "Finetune word embeddings? [False]")
    flags.DEFINE_bool("highway", True, "Use highway? [True]")
    flags.DEFINE_integer("highway_num_layers", 2, "highway num layers [2]")
    flags.DEFINE_bool("share_cnn_weights", True, "Share Char-CNN weights [True]")
    flags.DEFINE_bool(
        "share_lstm_weights",
        True,
        "Share pre-processing (phrase-level) LSTM weights [True]",
    )
    flags.DEFINE_float(
        "var_decay",
        float(var_decay),
        "Exponential moving average decay for variables [0.999]",
    )
    flags.DEFINE_string("classifier", "maxpool", "[maxpool, sumpool, default]")

    # Optimizations
    flags.DEFINE_bool("cluster", True, "Cluster data for faster training [False]")
    flags.DEFINE_bool("len_opt", True, "Length optimization? [False]")
    flags.DEFINE_bool(
        "cpu_opt", False, "CPU optimization? GPU computation can be slower [False]"
    )

    # Logging and saving options
    flags.DEFINE_boolean("progress", True, "Show progress? [True]")
    flags.DEFINE_integer("log_period", 100, "Log period [100]")
    flags.DEFINE_integer("eval_period", int(eval_period), "Eval period [1000]")
    flags.DEFINE_integer("save_period", int(save_period), "Save Period [1000]")
    flags.DEFINE_integer("max_to_keep", 20, "Max recent saves to keep [20]")
    flags.DEFINE_bool("dump_eval", True, "dump eval? [True]")
    flags.DEFINE_bool("dump_answer", False, "dump answer? [True]")
    flags.DEFINE_bool("vis", False, "output visualization numbers? [False]")
    flags.DEFINE_bool("dump_pickle", True, "Dump pickle instead of json? [True]")
    flags.DEFINE_float(
        "decay", 0.9, "Exponential moving average decay for logging values [0.9]"
    )

    # Thresholds for speed and less memory usage
    flags.DEFINE_integer("word_count_th", 10, "word count th [100]")
    flags.DEFINE_integer("char_count_th", 50, "char count th [500]")
    flags.DEFINE_integer("sent_size_th", int(sent_size_th), "sent size th [64]")
    flags.DEFINE_integer("num_sents_th", 1, "num sents th [8]")
    flags.DEFINE_integer("ques_size_th", int(ques_size_th), "ques size th [32]")
    flags.DEFINE_integer("word_size_th", 16, "word size th [16]")
    flags.DEFINE_integer("para_size_th", 256, "para size th [256]")

    # Advanced training options
    flags.DEFINE_bool("lower_word", True, "lower word [True]")
    flags.DEFINE_bool("squash", False, "squash the sentences into one? [False]")
    flags.DEFINE_bool("swap_memory", True, "swap memory? [True]")
    flags.DEFINE_string("data_filter", "max", "max | valid | semi [max]")
    flags.DEFINE_bool("use_glove_for_unk", True, "use glove for unk [False]")
    flags.DEFINE_bool(
        "known_if_glove", True, "consider as known if present in glove [False]"
    )
    flags.DEFINE_string("logit_func", "tri_linear", "logit func [tri_linear]")
    flags.DEFINE_string("answer_func", "linear", "answer logit func [linear]")
    flags.DEFINE_string("sh_logit_func", "tri_linear", "sh logit func [tri_linear]")

    # Ablation options
    flags.DEFINE_bool("use_char_emb", True, "use char emb? [True]")
    flags.DEFINE_bool("use_word_emb", True, "use word embedding? [True]")
    flags.DEFINE_bool("q2c_att", True, "question-to-context attention? [True]")
    flags.DEFINE_bool("c2q_att", True, "context-to-question attention? [True]")
    flags.DEFINE_bool("dynamic_att", False, "Dynamic attention [False]")

    def main(_):
        from basic.main import main as m

        config = flags.FLAGS
        config.out_dir = os.path.join(
            config.out_base_dir, config.model_name, str(config.run_id).zfill(2)
        )
        evaluator = m(config)

        """Generating metrics for the squad model"""
        if training_mode == "span":
            metrics = {
                "metrics": [
                    {
                        "name": "accuracy-score",
                        "numberValue": str(evaluator.acc),
                        "format": "RAW",
                    },
                    {
                        "name": "f1-score",
                        "numberValue": str(evaluator.f1),
                        "format": "RAW",
                    },
                ]
            }
        else:
            metrics = {
                "metrics": [
                    {
                        "name": "accuracy-score",
                        "numberValue": str(evaluator.acc),
                        "format": "RAW",
                    },
                    {
                        "name": "loss",
                        "numberValue": str(evaluator.loss),
                        "format": "RAW",
                    },
                ]
            }

        import json

        with open(mlpipeline_metrics_path, "w") as f:
            json.dump(metrics, f)

    tf.app.run(main)
