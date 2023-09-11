# Project variables
IMAGE_NAME ?= quay.io/rose/rose-ml-driver
PORT ?= 8081

build-image:
	@echo "Building Docker image..."
	podman build -t $(IMAGE_NAME) .

run-image:
	@echo "Running container image ..."
	podman run --rm \
		--network host \
		-it $(IMAGE_NAME) \
		--port $(PORT) \
		--driver /ml/driver.py
