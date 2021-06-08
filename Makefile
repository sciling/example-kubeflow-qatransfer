finalize:
	poetry update

pre-commit:
	poetry run pre-commit install

install: finalize
	pip install -r requirements.txt

tests: install
	./run_semeval_tests.sh
