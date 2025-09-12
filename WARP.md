# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

Project: NVIDIA Dynamo — high-throughput, low-latency distributed inference framework for LLMs. Core runtime is Rust; Python components wrap the runtime for engines like vLLM, SGLang, and TensorRT-LLM.

Key external services: etcd (service discovery) and NATS with JetStream (messaging, state snapshots). Use docker compose for local dev.

Common commands

- Bootstrap (Python + Rust bindings)
  - Create venv and install build tooling
    - curl -LsSf https://astral.sh/uv/install.sh | sh
    - uv venv .venv && source .venv/bin/activate
    - uv pip install pip maturin
  - Build Rust→Python bindings and install the wheel
    - pushd lib/bindings/python && maturin develop --uv && popd
    - uv pip install .
    - For editable-style development across split packages, set PYTHONPATH:
      - export PYTHONPATH="${PYTHONPATH}:$(pwd)/components/frontend/src:$(pwd)/components/planner/src:$(pwd)/components/backends/vllm/src:$(pwd)/components/backends/sglang/src:$(pwd)/components/backends/trtllm/src:$(pwd)/components/backends/llama_cpp/src:$(pwd)/components/backends/mocker/src"
  - Install engine extras (choose one or more)
    - uv pip install "ai-dynamo[vllm]"
    - uv pip install "ai-dynamo[sglang]"
    - uv pip install "ai-dynamo[trtllm]"
    - uv pip install "ai-dynamo[llama_cpp]"

- Required infra (local)
  - docker compose -f deploy/docker-compose.yml up -d
  - Exposes etcd and NATS for discovery and messaging.

- Run a local server + worker (quick start)
  - Frontend (OpenAI-compatible HTTP server with router)
    - python -m dynamo.frontend --http-port 8000 [--tls-cert-path cert.pem] [--tls-key-path key.pem]
  - Example worker (SGLang)
    - python -m dynamo.sglang.worker --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B --skip-tokenizer-init
  - Test request
    - curl localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"deepseek-ai/DeepSeek-R1-Distill-Llama-8B","messages":[{"role":"user","content":"Hello"}],"stream":false,"max_tokens":128}' | jq
  - Optional: KV-aware router mode
    - python -m dynamo.frontend --router-mode kv --http-port 8000
  - Logging level for Rust components
    - export DYN_LOG=debug  # same syntax as RUST_LOG

- Rust workspace (build/lint/test)
  - Build all crates
    - cargo build
  - Format and lint (match CI pre-merge-rust)
    - cargo fmt -- --check
    - cargo clippy --no-deps --all-targets -- -D warnings
  - Tests
    - cargo test --locked --all-targets
    - Doc tests
      - cargo doc --no-deps && cargo test --locked --doc

- Python linting/formatting (via pre-commit)
  - First time: pre-commit install
  - Run on all files
    - pre-commit run -a
  - Tools configured: ruff (with --fix), black, isort, flake8, codespell, clang-format (for C/C++/proto), basic file checks. Ruff/black/isort settings live in pyproject.toml.

- Tests (pytest)
  - Run all tests: pytest
  - By marker:
    - pytest -v -m e2e
    - pytest -v -m vllm
    - pytest -v -m "unit and not slow"
  - Single file / single test
    - pytest tests/path/to/test_file.py::TestClass::test_case
    - pytest -k "name_substring" -v
  - Local deps (outside container): ensure pytest-mypy and pytest-asyncio are installed
    - uv pip install pytest-mypy pytest-asyncio

- Docs
  - Local Sphinx build
    - uv pip install -r container/deps/requirements.docs.txt
    - make -C docs html
    - Output: docs/build/html/index.html
  - Helm and CRD docs helpers
    - make -C docs generate-helm-docs
    - make -C docs generate-crd-docs
  - Containerized docs (matches CI)
    - docker build -t docs-builder -f container/Dockerfile.docs .
    - CID=$(docker create docs-builder) && docker cp "$CID":/workspace/dynamo/docs/build/html dynamo-docs/ && docker rm "$CID"

- Containers (developer workflow)
  - Build images (framework-specific)
    - ./container/build.sh --framework vllm
    - ./container/build.sh --framework trtllm
    - ./container/build.sh --framework sglang
    - Add --target local-dev for user-mapped dev image; use --dry-run to inspect underlying docker commands.
  - Run images
    - ./container/run.sh --mount-workspace -it -- bash
    - Example test inside container
      - ./container/run.sh --mount-workspace -it -- pytest -v

- dynamo-run (Rust reference binary for local models and prototyping)
  - Build with features
    - cargo build -p dynamo-run                # CPU
    - cargo build -p dynamo-run --features metal  # macOS GPU
    - cargo build -p dynamo-run --features cuda   # Linux CUDA
  - Quick examples
    - dynamo-run Qwen/Qwen3-0.6B
    - echo 'Hi' | dynamo-run --context-length 4096 <model>
  - Distributed demo (requires etcd + NATS)
    - Node 1 (HTTP ingress): dynamo-run in=http out=auto
    - Node 2 (worker): dynamo-run in=dyn://llama3B.backend.generate out=mistralrs ~/llms/Llama-3.2-3B

Architecture overview (big picture)

- Distributed runtime (Rust, lib/runtime)
  - Core orchestration layer built around Namespace → Component → Endpoint.
  - Service discovery via etcd and transport via NATS:
    - Endpoints register under /services/{namespace}/{component}/{endpoint}-{lease_id} in etcd and a NATS service group {namespace}.{service}.{endpoint}.
    - Clients watch etcd, choose targets using pluggable strategies (random, round_robin, direct), then publish requests to NATS and receive results over a TCP stream.
  - Leases/cancellation tokens are used to clean up dynamic registrations; JetStream is leveraged by higher-level components for durable streams/snapshots.

- Frontend + Router (Python package backed by Rust runtime)
  - OpenAI-compatible HTTP server handles request validation and preprocessing (templates, tokenization), then routes to workers.
  - Router supports multiple strategies; KV-aware routing minimizes prefill by selecting workers with maximal prefix cache overlap, while balancing decode load.
  - KV-aware routing state
    - Workers emit KV block create/remove events (KVPublisher) to a JetStream topic.
    - Router’s KVIndexer maintains a global radix tree of prefix blocks (persisted via JetStream and NATS object store snapshots for fast recovery).
    - Active decode blocks are shared across router replicas optionally via --router-replica-sync.
    - Cost model: cost = overlap_weight * prefill_blocks + decode_blocks; optional temperature for softmax sampling.

- Disaggregated Serving (Prefill/Decode split)
  - Decode workers pre-allocate KV blocks in local GPU memory; when needed they enqueue prefill work.
  - Prefill workers pull work from a NATS consumer group, compute prefill, and use NIXL to write KV tensors directly into the decode worker’s allocated blocks (GPU→GPU over NVLink/PCIe) before decode continues.
  - This separation improves throughput per GPU and allows SLA-oriented tuning of TTFT vs ITL by scaling prefill vs decode fleets independently.

- KV Block Manager (KVBM)
  - Unified KV memory manager across tiers (HBM, pinned host, SSD, remote storage) with block lifecycle events, integrating with NIXL for registration/transfer. Enables larger effective context lengths and better TTFT/throughput under load.

- Engines and components
  - Engines live under components/backends/{vllm, sglang, trtllm, llama_cpp, mocker} and expose endpoints like generate. Python packages in components/* wrap Rust runtime and register services at startup.
  - Planner monitors metrics and scales prefill/decode pools based on demand; metrics and auxiliary components reside under components/ and docs/runtime.

CI signals and how to match them locally

- Python pre-commit runs on PRs: run pre-commit run -a locally to mirror checks.
- Rust pre-merge job runs fmt, clippy (deny warnings), builds+tests across workspace roots (., lib/bindings/python, lib/runtime/examples). Use the exact commands listed above.
- Documentation build uses container/Dockerfile.docs; prefer docker-based build for parity when validating doc changes.

Reference docs in repo

- docs/architecture/* — high-level architecture, disaggregated serving, KV routing, KVBM, distributed runtime
- docs/guides/dynamo_run.md — usage and development of the dynamo-run binary
- components/backends/* — backend-specific capabilities and deployment notes
- container/README.md — images and run.sh usage matrix; build/run examples
- tests/README.md — test markers, common invocations, local setup

Notes

- No CLAUDE, Cursor, or Copilot rules were found in this repo at the time of writing.
- For GPU engine backends (vLLM/TRT-LLM), consult each backend’s README for model- and CUDA-specific flags; ensure compatible container/toolchain versions as documented in README.md.
