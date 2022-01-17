.PHONY: check
check:
	mypy --ignore-missing-imports cluster_video.py

.PHONY: format
format:
	black *.py

.PHONY: install
install:
	pip install -r requirements.txt -r dev-requirements.txt
