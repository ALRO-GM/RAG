# TORCH=cuda runs with the CUDA build of PyTorch (after install-cuda).
TORCH       ?= cpu
ifeq ($(TORCH),cuda)
UV_GROUPS   := --group cuda --no-group cpu
else
UV_GROUPS   :=
endif
PYTHON      := uv run $(UV_GROUPS) python
ARGS        ?= --help
K           ?= 10
SCOPE       ?= UnansweredQuestions
DATASETS    := docs code
SUITE       ?= tune_all
SERVE_ARGS  ?=

# Server that serves the transcriber's pages (the domain resolves to it).
DEPLOY_TARGET ?= ubuntu@rag.peer4u.app
REMOTE_DIR    := /opt/ragputin
# The workstation's own key for the SSH reverse forward (make tunnel-key).
TUNNEL_KEY    ?= $(HOME)/.ssh/ragputin_tunnel

.PHONY: install run debug clean lint lint-strict test test-slow index search \
	evaluate answer serve experiment tune analyze skills-demo \
	install-cuda system system-benchmark study coverage ci site \
	infra-up infra-proxy tunnel-key transcriber-serve transcriber-test

install:
	uv sync

## install-cuda: PyTorch with CUDA 12.6 (needs the NVIDIA driver loaded).
## Afterwards use `make <target> TORCH=cuda`, or `uv run --group cuda
## --no-group cpu python -m src ...`, or export UV_NO_SYNC=1.
install-cuda:
	uv sync --group cuda --no-group cpu
	uv run --group cuda --no-group cpu python -m src system

## system: hardware found, runtime plan and why a GPU is not used.
system:
	$(PYTHON) -m src system

## system-benchmark: measure CPU plans against the GPU; the run is
## appended to experiments/hardware/runtime.jsonl.
system-benchmark:
	$(PYTHON) -m src system --benchmark

## run: index the corpus and search both public datasets.
run: index search

## debug: run the CLI under pdb, e.g. make debug ARGS='search "lora"'.
debug:
	$(PYTHON) -m pdb -m src $(ARGS)

index:
	$(PYTHON) -m src index --max_chunk_size 2000

search:
	@for d in $(DATASETS); do \
		$(PYTHON) -m src search_dataset \
			--dataset_path data/datasets/$(SCOPE)/dataset_$${d}_public.json \
			--k $(K) \
			--save_directory data/output/search_results/$(SCOPE) || exit 1; \
	done

## evaluate: score the search results with our own recall@k.
evaluate:
	@for d in $(DATASETS); do \
		echo "== $$d"; \
		$(PYTHON) -m src evaluate \
			--student_search_results_path \
			data/output/search_results/$(SCOPE)/dataset_$${d}_public.json \
			--dataset_path data/datasets/AnsweredQuestions/dataset_$${d}_public.json \
			--k $(K); \
	done

answer:
	@for d in $(DATASETS); do \
		$(PYTHON) -m src answer_dataset \
			--student_search_results_path \
			data/output/search_results/$(SCOPE)/dataset_$${d}_public.json \
			--save_directory data/output/search_results_and_answer/$(SCOPE) \
			|| exit 1; \
	done

serve:
	$(PYTHON) -m src serve

## experiment: run a lab suite, e.g. make experiment SUITE=search_priors.
experiment:
	$(PYTHON) -m src lab experiment --suite $(SUITE)

## study: generalist study of a suite on the GPU: 42 lab + 7 public
## benchmarks + combined report (experiments/general/SUITE.md).
## e.g. make study SUITE=general_models TORCH=cuda
GENERAL_DATASETS ?= scifact,nfcorpus,arguana,scidocs,fiqa,coir:cosqa,coir:stackoverflow-qa
study:
	$(PYTHON) -m src lab experiment --suite $(SUITE)
	$(PYTHON) -m src benchmark --suite $(SUITE) \
		--datasets $(GENERAL_DATASETS) \
		--output experiments/general/benchmarks.jsonl --clean_indexes
	$(PYTHON) -m src general --suite $(SUITE)

## tune: cross-validated tuning of the factors of a suite.
tune:
	$(PYTHON) -m src lab tune --suite $(SUITE)

## analyze: statistics and recommended profiles -> experiments/REPORT.md.
analyze:
	$(PYTHON) -m src lab analyze

## skills-demo: evidence retrieval on the synthetic Spanish session.
skills-demo:
	$(PYTHON) -m src skills interaction \
		--transcript examples/skills/transcripts/demo_es.txt
	$(PYTHON) -m src skills evidence \
		--transcript examples/skills/transcripts/demo_es.txt \
		--participant Pablo --competency LISTEN

test:
	uv run pytest -q

test-slow:
	uv run pytest -q -m slow

## transcriber-serve: the transcriber API, the tunnel to the server and
## the join link with a QR code, e.g. make transcriber-serve TORCH=cuda
## SERVE_ARGS='--keep-code' (or SERVE_ARGS='--no-tunnel' for local use).
transcriber-serve:
	$(PYTHON) -m transcriber serve $(SERVE_ARGS)

## transcriber-test: the fast transcriber tests (fake models, no network).
transcriber-test:
	uv run pytest -q tests/unit/transcriber \
		tests/integration/transcriber

## coverage: fast tests with branch coverage (fails under the gate).
coverage:
	uv run pytest -q --cov --cov-report=term --cov-report=html

## ci: what GitHub Actions runs, locally.
ci: lint-strict coverage site

## site: build the transcriber's static pages into dist/site, with
## version.json (what CI deploys to the server).
site:
	uv run --no-sync python -m transcriber.adapters.inbound.bundle \
		--out dist/site --commit "$$(git rev-parse --short HEAD)"

## infra-up: upload infra/ and (re)start the RAGPutin web container.
## Also makes /opt/ragputin/run (mode 750, owner ubuntu), where the
## tunnel's socket appears, and installs the tunnel key's forced command.
infra-up:
	ssh $(DEPLOY_TARGET) 'mkdir -p $(REMOTE_DIR)/infra $(REMOTE_DIR)/pruebas && install -d -m 750 $(REMOTE_DIR)/run'
	rsync -rlz infra/Caddyfile infra/compose.yml $(DEPLOY_TARGET):$(REMOTE_DIR)/infra/
	rsync -pz --chmod=F755 infra/tunnel-clean.sh $(DEPLOY_TARGET):$(REMOTE_DIR)/infra/
	ssh $(DEPLOY_TARGET) 'cd $(REMOTE_DIR)/infra && docker compose up -d && docker compose ps'

## infra-proxy: install the rag.peer4u.app block in the Peer4U Caddy
## (validated first, rolled back on failure).
infra-proxy:
	infra/install-proxy.sh $(DEPLOY_TARGET)

## tunnel-key: make the workstation's tunnel key if it is missing and
## print the authorized_keys line to add on the server by hand. It
## installs nothing anywhere.
tunnel-key:
	@test -f $(TUNNEL_KEY) || ssh-keygen -q -t ed25519 -N "" -C ragputin-tunnel -f $(TUNNEL_KEY)
	@echo "Add this line to ~/.ssh/authorized_keys of $(DEPLOY_TARGET):"
	@echo "restrict,port-forwarding,command=\"$(REMOTE_DIR)/infra/tunnel-clean.sh\" $$(cat $(TUNNEL_KEY).pub)"

clean:
	find . -name __pycache__ -type d -not -path "./.venv/*" -prune \
		-exec rm -r {} +
	rm -rf .mypy_cache .pytest_cache .coverage coverage.xml htmlcov dist

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores \
		--ignore-missing-imports --disallow-untyped-defs \
		--check-untyped-defs

lint-strict:
	uv run flake8 .
	uv run mypy . --strict
