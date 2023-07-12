#  Copyright 2021 The HuggingFace Team. All rights reserved.
#  Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
SHELL := /bin/bash
CURRENT_DIR = $(shell pwd)
DEFAULT_CLONE_URL := https://github.com/huggingface/optimum-graphcore.git
# If CLONE_URL is empty, revert to DEFAULT_CLONE_URL
REAL_CLONE_URL = $(if $(CLONE_URL),$(CLONE_URL),$(DEFAULT_CLONE_URL))
DEFAULT_CLONE_NAME := optimum-graphcore
# If CLONE_NAME is empty, revert to DEFAULT_CLONE_NAME
REAL_CLONE_NAME = $(if $(CLONE_NAME),$(CLONE_NAME),$(DEFAULT_CLONE_NAME))

.PHONY:	style test

check_dirs := examples tests optimum

# Run code quality checks
style_check:
	black --check $(check_dirs)
	ruff $(check_dirs)

style:
	black $(check_dirs)
	ruff $(check_dirs) --fix

# Run tests for the library
test:
	python -m pytest tests

# Utilities to release to PyPi
build_dist_install_tools:
	pip install build
	pip install twine

build_dist:
	rm -fr build
	rm -fr dist
	python -m build

pypi_upload: build_dist
	python -m twine upload dist/*

build_doc_docker_image:
	docker build -t doc_maker --build-arg commit_sha=$(COMMIT_SHA_SUBPACKAGE) --build-arg clone_url=$(REAL_CLONE_URL) --build-arg clone_name=$(REAL_CLONE_NAME) ./docs

doc: build_doc_docker_image
	@test -n "$(BUILD_DIR)" || (echo "BUILD_DIR is empty." ; exit 1)
	@test -n "$(VERSION)" || (echo "VERSION is empty." ; exit 1)
	docker run -v $(CURRENT_DIR):/doc_folder --workdir=/doc_folder doc_maker \
	doc-builder build optimum.graphcore /$(REAL_CLONE_NAME)/docs/source/ \
		--build_dir $(BUILD_DIR) \
		--version $(VERSION) \
		--version_tag_suffix "" \
		--html \
		--clean


# make custom_ops
# Builds the group_quantize_decompress custom ops

CXX ?= g++
OUT ?= build/custom_ops.so
OBJDIR ?= $(dir $(OUT))obj

CXXFLAGS = -Wall -Wno-sign-compare -std=c++17 -O2 -g -fPIC -DONNX_NAMESPACE=onnx
LIBS = -lpoplar -lpopart -lpopops -lpoplin -lpopnn -lpoputil -lpoprand

OBJECTS = $(OBJDIR)/group_quantize_decompress.o $(OBJDIR)/group_quantize_decompressx.o

# Rules

custom_ops: $(OUT)

$(OBJECTS): $(OBJDIR)/%.o: optimum/graphcore/custom_ops/group_quantize_decompress/%.cpp
	@mkdir -p $(@D)
	$(CXX) -c $(CXXFLAGS) $< -o $@

$(OUT): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -shared $^ -o $@ -Wl,--no-undefined $(LIBS)