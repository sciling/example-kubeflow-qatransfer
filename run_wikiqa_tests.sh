docker run --rm -ti -v $PWD:/tmp/qa/ sciling/tensorflow:0.12.0-py3 bash -c '
cd /tmp/qa/;
pip install -r requirements.txt;
python -m unittest discover /tmp/qa/tests/wikiqa'