.PHONY: check
check:
	mypy --ignore-missing-imports cluster_video.py

.PHONY: format
format:
	black cluster_video.py