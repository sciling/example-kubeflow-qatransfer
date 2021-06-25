docker run --rm -ti -v $PWD:/tmp/qa/ sciling/tensorflow:0.12.0-gpu-py3 bash -c '
cd /tmp/qa/;
pip install -r requirements.txt;
python -m unittest $PWD/tests/semeval/test_semeval_{01_prepro,02_train,03_generate_files,04_test}.py'
