import os
import pathlib
import sys
import tempfile
import unittest


sys.path.append("..")


DATA_DIR = "%s/../data/test/squad_files" % pathlib.Path(__file__).parent.absolute()
WORK_DIR = tempfile.mkdtemp()


class TestAll(unittest.TestCase):
    def test_train(self):
        from src.squad.train import train

        squad_path = DATA_DIR
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
        model_path = tempfile.mkdtemp()
        try:
            from multiprocessing import Process

            args = (
                squad_path,
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
                model_path,
            )
            p = Process(target=train, args=args)
            p.start()
            p.join()

        except SystemExit:
            print("Finished successfully!")

        # Check model directory has all files
        self.assertIn("out", os.listdir(model_path))
        self.assertIn("squad", os.listdir(model_path + "/out"))


if __name__ == "__main__":
    unittest.main()
