finalize:
	echo 'Does not have finalize'

pre-commit:
	echo 'Does not have pre-commit'

install: finalize
	pip install -r requirements.txt

tests: install
	python -m unittest /tmp/qa/tests/semeval/test_semeval_{prepro,train,generate_files,test}.py
