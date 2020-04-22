#!make
PROJECT_VERSION := $(shell python setup.py --version)

SHELL := /bin/bash
PACKAGE := pyhrp

.PHONY: help build test teamcity graph doc tag all_commit clean


.DEFAULT: help

help:
	@echo "make build"
	@echo "       Build the docker image."
	@echo "make test"
	@echo "       Build the docker image for testing and run them."
	@echo "make teamcity"
	@echo "       Run tests, build a dependency graph and construct the documentation."
	#@echo "make jupyter"
	#@echo "       Start the Jupyter server."
	@echo "make graph"
	@echo "       Build a dependency graph."
	@echo "make doc"
	@echo "       Construct the documentation."
	@echo "make tag"
	@echo "       Make a tag on Github."



build:
	docker-compose build pyhrp

test:
	mkdir -p artifacts
	docker-compose -f docker-compose.test.yml run sut

teamcity: test graph

graph: test
	mkdir -p artifacts/graph

	docker run --rm --mount type=bind,source=${PWD}/${PACKAGE},target=/pyan/${PACKAGE},readonly \
		   tschm/pyan:latest python pyan.py ${PACKAGE}/**/*.py -V --uses --defines --colored --dot --nested-groups > graph.dot

	# remove all the private nodes...
	grep -vE "____" graph.dot > graph2.dot

	docker run --rm -v ${PWD}/graph2.dot:/pyan/graph.dot:ro \
		   tschm/pyan:latest dot -Tsvg /pyan/graph.dot > artifacts/graph/graph.svg

	rm graph.dot graph2.dot

tag: test
	git tag -a ${PROJECT_VERSION} -m "new tag"
	git push --tags

clean:
	docker-compose -f docker-compose.yml down -v --rmi all --remove-orphans
	docker-compose -f docker-compose.test.yml down -v --rmi all --remove-orphans

pypi: #tag
	python setup.py sdist
	twine check dist/*
