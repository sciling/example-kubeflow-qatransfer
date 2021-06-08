import sys
import tempfile
import unittest


sys.path.append("..")

WORK_DIR = '/tmp/wikiqa-tests'


class TestAll(unittest.TestCase):
    def test_eval(self):
        import json

        from src.wikiqa.wikiqa_evaluate import wikiqa_evaluate

        start_step = "2"
        end_step = "2"
        eval_period = "10"
        run_ids = "00"
        mlpipeline_metrics_path = tempfile.NamedTemporaryFile()

        wikiqa_evaluate(
            WORK_DIR,
            WORK_DIR,
            start_step,
            end_step,
            eval_period,
            run_ids,
            mlpipeline_metrics_path.name,
            )

        with open(mlpipeline_metrics_path.name, "r") as f:
            metrics = json.load(f)
        self.assertIn("metrics", list(metrics.keys()))
        self.assertEqual(3, len(list(metrics["metrics"])))
        self.assertEqual("MAP_for_run_2", metrics["metrics"][0]["name"])
        self.assertEqual("MRR_for_run_2", metrics["metrics"][1]["name"])
        self.assertEqual("P1_for_run_2", metrics["metrics"][2]["name"])
        mlpipeline_metrics_path.close()


if __name__ == "__main__":
    unittest.main()
