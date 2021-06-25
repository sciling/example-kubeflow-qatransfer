finalize:
	poetry update

pre-commit:
	poetry run pre-commit install

install: finalize
	poetry install

tests: install
	# tests files need to follow an alphanumeric order to success. That is because of the dependencies between tests.
	poetry run coverage run -m pytest -p no:sugar -s -q tests/
