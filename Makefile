finalize:
	echo 'Does not have finalize'

pre-commit:
	poetry run pre-commit install

install: finalize
	pip install -r requirements.txt

tests: install
	python -m unittest ./tests/semeval/test_semeval_{prepro,train,generate_files,test}.py
