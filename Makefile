.PHONY: init tensorboard kill-tensorboard

ROOT_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

## Create virtual environment and install dependencies
init:
	conda env update --file environment.yaml

## Start the tensorboard server
tensorboard:
	tensorboard --logdir=$(ROOT_DIR)/artifacts/logs/board/ --port=6006 --load_fast=false

## Kill tensorboard server
kill-tensorboard:
	kill $(ps -e | grep 'tensorboard' | awk '{print $1}')

## Lint the code
lint:
	(\
	python -m flake8 src --max-line-length=120 && \
	python -m black src --check --diff --color -l 120 \
	)

## Format the code
format:
	(\
	python -m black src -l 120 \
	)

## Delete compiled Python files
clean-files:
	find . | grep -E "build$|\/__pycache__$|\.pyc$|\.pyo$|\.egg-info$|\.ipynb_checkpoints" | xargs rm -rf || echo "Already clean"

DEFAULT_GOAL := help
.PHONY: help
help:
	@echo "\n$$(tput bold)Available rules:$$(tput sgr0)\n"
	@awk '/^##/{c=substr($$0,3);next}c&&/^[[:alpha:]][[:alnum:]_-]+:/{print substr($$1,1,index($$1,":")),c}1{c=0}' $(MAKEFILE_LIST) | column -s: -t
