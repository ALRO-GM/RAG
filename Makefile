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

# Server that hosts the results site (the domain resolves to it).
DEPLOY_TARGET ?= ubuntu@rag.peer4u.app
REMOTE_DIR    := /opt/ragputin

.PHONY: install run debug clean lint lint-strict test test-slow index search \
	evaluate answer serve experiment tune analyze skills-demo \
	install-cuda system system-benchmark study coverage ci site site-serve \
	deploy-site infra-up infra-proxy

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
## appended to experiments/hardware/runtime.jsonl (shown on the site).
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
		--output experiments/general/benchmarks.jsonl
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

## coverage: fast tests with branch coverage (fails under the gate).
coverage:
	uv run pytest -q --cov --cov-report=term --cov-report=html

## ci: what GitHub Actions runs, locally.
ci: lint-strict coverage site

## site: build the static results site into dist/site.
site:
	uv run --no-sync python -m results_site build --out dist/site

## site-serve: preview the site at http://localhost:8765.
site-serve: site
	cd dist/site && python3 -m http.server 8765

## deploy-site: manual upload (CI does this on every push to main).
deploy-site: site
	rsync -rlz --delete --delay-updates --chmod=D755,F644 dist/site/ \
		$(DEPLOY_TARGET):$(REMOTE_DIR)/pruebas/

## infra-up: upload infra/ and (re)start the RAGPutin web container.
infra-up:
	ssh $(DEPLOY_TARGET) 'mkdir -p $(REMOTE_DIR)/infra $(REMOTE_DIR)/pruebas'
	rsync -rlz infra/Caddyfile infra/compose.yml $(DEPLOY_TARGET):$(REMOTE_DIR)/infra/
	ssh $(DEPLOY_TARGET) 'cd $(REMOTE_DIR)/infra && docker compose up -d && docker compose ps'

## infra-proxy: install the rag.peer4u.app block in the Peer4U Caddy
## (validated first, rolled back on failure).
infra-proxy:
	infra/install-proxy.sh $(DEPLOY_TARGET)

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
