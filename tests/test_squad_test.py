import pathlib
import sys
import tempfile
import unittest


sys.path.append("..")


DATA_DIR = "%s/../data/test/squad_files" % pathlib.Path(__file__).parent.absolute()
WORK_DIR = tempfile.mkdtemp()


class TestAll(unittest.TestCase):
    def test_test(self):
        import json

        from src.squad.test import test

        prepro_dir = DATA_DIR
        prev_model_dir = DATA_DIR
        sent_size_th = "10"
        ques_size_th = "10"
        num_epochs = "1"
        num_steps = "1"
        eval_period = "1"
        save_period = "1"
        learning_rate = "0.5"
        batch_size = "60"
        hidden_size = "100"
        var_decay = "0.999"
        mlpipeline_metrics_path = tempfile.NamedTemporaryFile()
        model_dir = tempfile.mkdtemp()
        try:
            from multiprocessing import Process

            args = (
                prepro_dir,
                prev_model_dir,
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
                mlpipeline_metrics_path.name,
                model_dir,
            )
            p = Process(target=test, args=args)
            p.start()
            p.join()
        except SystemExit:
            print("Finished successfully!")

        with open(mlpipeline_metrics_path.name, "r") as f:
            metrics = json.load(f)
        self.assertIn("metrics", list(metrics.keys()))
        self.assertEqual(2, len(list(metrics["metrics"])))
        self.assertEqual("accuracy-score", metrics["metrics"][0]["name"])
        self.assertEqual("f1-score", metrics["metrics"][1]["name"])
        mlpipeline_metrics_path.close()


if __name__ == "__main__":
    unittest.main()
