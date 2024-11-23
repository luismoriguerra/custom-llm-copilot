venv:
	python3 -m venv venv

install: venv
	. venv/bin/activate && pip install -r requirements.txt

dev: install
	. venv/bin/activate && python3 server.py

clean:
	rm -rf venv
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete


vectordb:
	bash standalone_embed.sh start

initdb:
	. venv/bin/activate && python3 codeIngestion.py

.PHONY: venv install dev clean
