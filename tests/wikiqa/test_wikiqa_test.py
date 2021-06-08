import sys
import tempfile
import unittest


sys.path.append("..")

WORK_DIR = '/tmp/wikiqa-tests'


class TestAll(unittest.TestCase):
    def test_test(self):
        import json
        from src.wikiqa.wikiqa_test import wikiqa_test

        shared_path = "/save/out/squad/basic-class/00/shared.json"
        run_id = "00"
        prepro_dir = WORK_DIR
        prev_model_dir = WORK_DIR
        sent_size_th = "10"
        ques_size_th = "10"
        num_epochs = "1"
        num_steps = "1"
        eval_period = "1"
        save_period = "1"
        mlpipeline_metrics_path = tempfile.NamedTemporaryFile()
        model_dir = tempfile.mkdtemp()
        try:
            from multiprocessing import Process

            args = (
                WORK_DIR,
                prepro_dir,
                prev_model_dir,
                shared_path,
                run_id,
                sent_size_th,
                ques_size_th,
                num_epochs,
                num_steps,
                eval_period,
                save_period,
                mlpipeline_metrics_path.name,
                model_dir,
                )
            p = Process(target=wikiqa_test, args=args)
            p.start()
            p.join()
        except SystemExit:
            print("Finished successfully!")

        with open(mlpipeline_metrics_path.name, "r") as f:
            metrics = json.load(f)
        self.assertIn("metrics", list(metrics.keys()))
        self.assertEqual(2, len(list(metrics["metrics"])))
        self.assertEqual("accuracy-score", metrics["metrics"][0]["name"])
        self.assertEqual("loss", metrics["metrics"][1]["name"])
        mlpipeline_metrics_path.close()


if __name__ == "__main__":
    unittest.main()
