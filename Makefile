# face-recognition CGP — standard operations.
#
# Targets:
#   make build      Pack + verify the patch via cpm (keeps tools exec bits)
#   make info       Print patch metadata from cognitive.json
#   make verify     Verify the packed archive integrity
#   make install    Build then install the .cgp locally
#   make clean      Remove build artifacts (*.cgp, out/)
#   make ci         Reproduce the GitHub CI build locally (requires docker)
#
# Override CPM to point at a specific cpm binary, e.g.:
#   make build CPM=/tmp/lab/cpm/build/bin/cpm

CPM ?= cpm
NAME ?= $(shell jq -r .name cognitive.json 2>/dev/null || echo face-recognition)
VERSION ?= $(shell jq -r .version cognitive.json 2>/dev/null || echo 0.1.0)
OUT = $(NAME)-$(VERSION).cgp

.PHONY: all build info verify install clean ci

all: build

build:
	$(CPM) pack --bin tools

info:
	$(CPM) info $(NAME) --json --manifest cognitive.json

verify:
	$(CPM) verify $(OUT)

install: build
	$(CPM) install $(OUT)

ci:
	docker build -t cgp-ci:local -f .github/docker/Dockerfile.ci .
	rm -rf out
	id=$$(docker create cgp-ci:local); \
	docker cp $$id:/out ./out; \
	docker rm $$id
	@echo "CI artifacts in ./out/"

clean:
	rm -f *.cgp
	rm -rf out
