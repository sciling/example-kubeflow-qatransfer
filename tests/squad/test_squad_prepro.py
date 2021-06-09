import os
import pathlib
import sys
import unittest


sys.path.append("..")


DATA_DIR = "%s/../../data/test" % pathlib.Path(__file__).parent.absolute()
WORK_DIR = '/tmp/squad-tests'


class TestAll(unittest.TestCase):
    def test_preprospan(self):
        from src.squad.prepro import prepro_basic
        import shutil
        if not os.path.exists(WORK_DIR + '/data/glove'):
            shutil.copytree(DATA_DIR + '/glove', WORK_DIR + '/data/glove')
        if not os.path.exists(WORK_DIR + '/data/squad'):
            shutil.copytree(DATA_DIR + '/data/squad', WORK_DIR + '/data/squad')
        squad_path = WORK_DIR
        dataset_path = WORK_DIR
        train_ratio = 0.9
        glove_vec_size = 100
        mode = "full"
        tokenizer = "PTB"
        url = "vision-server2.corp.ai2"
        port = 8000
        prepro_basic(
            dataset_path,
            train_ratio,
            glove_vec_size,
            mode,
            tokenizer,
            url,
            port,
            squad_path,
            )

        # Check model directory has all files
        self.assertIn("data_dev.json", os.listdir(squad_path + "/squad"))
        self.assertIn("data_test.json", os.listdir(squad_path + "/squad"))
        self.assertIn("data_train.json", os.listdir(squad_path + "/squad"))
        self.assertIn("shared_dev.json", os.listdir(squad_path + "/squad"))
        self.assertIn("shared_test.json", os.listdir(squad_path + "/squad"))
        self.assertIn("shared_train.json", os.listdir(squad_path + "/squad"))

    def test_convert2class(self):
        from src.squad.prepro import convert2class
        import os
        class_dir = WORK_DIR
        convert2class(DATA_DIR, class_dir)

        self.assertIn("data", os.listdir(class_dir))
        self.assertIn("squad-class", os.listdir(class_dir + "/data"))
        self.assertIn("train-v1.1.json", os.listdir(class_dir + "/data/squad-class"))
        self.assertIn("dev-v1.1.json", os.listdir(class_dir + "/data/squad-class"))

    def test_preproclass(self):
        from src.squad.prepro import prepro_class
        import shutil
        if not os.path.exists(WORK_DIR + '/data/glove'):
            shutil.copytree(DATA_DIR + '/glove', WORK_DIR + '/data/glove')
        squad_path = WORK_DIR
        dataset_path = WORK_DIR
        train_ratio = 0.9
        glove_vec_size = 100
        mode = "full"
        tokenizer = "PTB"
        url = "vision-server2.corp.ai2"
        port = 8000
        prepro_class(
            dataset_path,
            WORK_DIR,
            train_ratio,
            glove_vec_size,
            mode,
            tokenizer,
            url,
            port,
            squad_path,
            )

        # Check model directory has all files
        self.assertIn("data_dev.json", os.listdir(squad_path + "/squad-class"))
        self.assertIn("data_test.json", os.listdir(squad_path + "/squad-class"))
        self.assertIn("data_train.json", os.listdir(squad_path + "/squad-class"))
        self.assertIn("shared_dev.json", os.listdir(squad_path + "/squad-class"))
        self.assertIn("shared_test.json", os.listdir(squad_path + "/squad-class"))
        self.assertIn("shared_train.json", os.listdir(squad_path + "/squad-class"))


if __name__ == "__main__":
    unittest.main()
