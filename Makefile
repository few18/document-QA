setup:
	python3 -m venv env

install:
	pip install -r requirements.txt

lint:
	pylint --disable=R,C **/*.py

format:
	isort **/*.py
	black **/*.py

all: format lint

