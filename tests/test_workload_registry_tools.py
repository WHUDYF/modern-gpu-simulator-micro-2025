import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.generate_source_registry import (
    build_source_registry,
    infer_clone_mode,
    parse_clone_status,
)
from scripts.generate_workload_registry import (
    build_workload_registry,
    discover_workloads_for_source,
    slugify_workload_part,
)


def init_git_repo(path: Path) -> str:
    path.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=path, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=path, check=True)
    subprocess.run(["git", "config", "commit.gpgsign", "false"], cwd=path, check=True)
    (path / "README.md").write_text("# fixture\n")
    subprocess.run(["git", "add", "README.md"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-m", "fixture"], cwd=path, check=True, stdout=subprocess.DEVNULL)
    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=path, text=True).strip()


def test_parse_clone_status_reads_tsv_rows(tmp_path):
    status = tmp_path / "clone_status.tsv"
    status.write_text(
        "name\tstatus\tcommit\tpath\turl\n"
        "gpu-rodinia\texists\tabc123\t/tmp/gpu-rodinia\thttps://example/rodinia.git\n"
    )

    rows = parse_clone_status(status)

    assert rows == [
        {
            "name": "gpu-rodinia",
            "status": "exists",
            "commit": "abc123",
            "path": "/tmp/gpu-rodinia",
            "url": "https://example/rodinia.git",
        }
    ]


def test_infer_clone_mode_detects_sparse_checkout(tmp_path):
    repo = tmp_path / "repo"
    init_git_repo(repo)
    subprocess.run(["git", "config", "core.sparseCheckout", "true"], cwd=repo, check=True)

    assert infer_clone_mode(repo) == "sparse_partial"


def test_build_source_registry_uses_local_git_commit(tmp_path):
    source = tmp_path / "sources" / "gpu-rodinia"
    commit = init_git_repo(source)
    status = tmp_path / "clone_status.tsv"
    status.write_text(
        "name\tstatus\tcommit\tpath\turl\n"
        f"gpu-rodinia\texists\told\t{source}\thttps://example/rodinia.git\n"
    )

    registry = build_source_registry(status)

    assert registry["schema_version"] == "source_registry_v1"
    assert registry["sources"][0]["source_id"] == "gpu-rodinia"
    assert registry["sources"][0]["commit"] == commit
    assert registry["sources"][0]["availability_status"] == "source_available"


def test_build_source_registry_accepts_git_worktree_with_git_file(tmp_path):
    main = tmp_path / "main"
    commit = init_git_repo(main)
    worktree = tmp_path / "worktree"
    subprocess.run(["git", "worktree", "add", str(worktree)], cwd=main, check=True, stdout=subprocess.DEVNULL)
    status = tmp_path / "clone_status.tsv"
    status.write_text(
        "name\tstatus\tcommit\tpath\turl\n"
        f"gpu-rodinia\texists\told\t{worktree}\thttps://example/rodinia.git\n"
    )

    registry = build_source_registry(status)

    assert (worktree / ".git").is_file()
    assert registry["sources"][0]["commit"] == commit
    assert registry["sources"][0]["clone_mode"] == "shallow_or_full"
    assert registry["sources"][0]["availability_status"] == "source_available"


def test_cli_generated_at_makes_artifacts_deterministic(tmp_path):
    source = tmp_path / "sources" / "gpu-rodinia"
    init_git_repo(source)
    root = tmp_path / "workloads"
    root.mkdir()
    (root / "clone_status.tsv").write_text(
        "name\tstatus\tcommit\tpath\turl\n"
        f"gpu-rodinia\texists\told\t{source}\thttps://example/rodinia.git\n"
    )
    output = tmp_path / "registry"

    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "generate_source_registry.py"),
            "--root",
            str(root),
            "--output-dir",
            str(output),
            "--generated-at",
            "2026-05-11T00:00:00+00:00",
        ],
        check=True,
    )

    first_json = (output / "source_registry.json").read_text()
    first_md = (output / "source_registry.md").read_text()
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "generate_source_registry.py"),
            "--root",
            str(root),
            "--output-dir",
            str(output),
            "--generated-at",
            "2026-05-11T00:00:00+00:00",
        ],
        check=True,
    )

    assert (output / "source_registry.json").read_text() == first_json
    assert (output / "source_registry.md").read_text() == first_md
    assert json.loads(first_json)["generated_at"] == "2026-05-11T00:00:00+00:00"


def test_discover_workloads_for_gpu_rodinia_cuda_dirs(tmp_path):
    root = tmp_path / "gpu-rodinia"
    (root / "cuda" / "bfs").mkdir(parents=True)
    (root / "cuda" / "hotspot").mkdir(parents=True)

    workloads = discover_workloads_for_source("gpu-rodinia", root)
    ids = {item["workload_id"] for item in workloads}

    assert "gpu-rodinia_bfs" in ids
    assert "gpu-rodinia_hotspot" in ids
    assert all(item["source_id"] == "gpu-rodinia" for item in workloads)


def test_discover_workloads_for_full_network_source_uses_curated_candidates(tmp_path):
    root = tmp_path / "mlperf-inference"
    root.mkdir()

    workloads = discover_workloads_for_source("mlperf-inference", root)
    ids = {item["workload_id"] for item in workloads}

    assert "mlperf-inference_bert" in ids
    assert "mlperf-inference_resnet50" in ids
    assert all(item["workload_family"] == "full_network" for item in workloads)


def test_discover_workloads_for_gpu_parboil_benchmark_src_dirs(tmp_path):
    root = tmp_path / "gpu-parboil"
    for name in ["bfs", "histo", "sgemm", "spmv"]:
        (root / "benchmarks" / name / "src").mkdir(parents=True)
    (root / "benchmarks" / "no-src").mkdir(parents=True)

    workloads = discover_workloads_for_source("gpu-parboil", root)
    ids = {item["workload_id"] for item in workloads}
    paths = {item["workload_id"]: item["relative_path"] for item in workloads}

    assert {"gpu-parboil_bfs", "gpu-parboil_histo", "gpu-parboil_sgemm", "gpu-parboil_spmv"} <= ids
    assert "gpu-parboil_no-src" not in ids
    assert paths["gpu-parboil_bfs"] == "benchmarks/bfs/src"


def test_discover_workloads_for_hecbench_uses_curated_cuda_candidates(tmp_path):
    root = tmp_path / "hecbench"
    for name in ["bfs-cuda", "bfs-hip", "sgemm-cuda", "sgemm-omp", "attention-cuda"]:
        (root / "src" / name).mkdir(parents=True)

    workloads = discover_workloads_for_source("hecbench", root)
    ids = {item["workload_id"] for item in workloads}

    assert {"hecbench_bfs", "hecbench_sgemm", "hecbench_attention"} <= ids
    assert "hecbench_bfs-hip" not in ids
    assert len(workloads) < 20
    assert all(item["relative_path"].endswith("-cuda") for item in workloads)


def test_slugify_workload_part_uses_portable_lowercase_charset():
    assert slugify_workload_part("B+Tree CUDA!") == "b-tree-cuda"
    assert slugify_workload_part("A___B...C") == "a___b...c"
    assert slugify_workload_part("---Bad/Name---") == "bad-name"


def test_build_workload_registry_detects_duplicate_normalized_ids(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    (first / "cuda" / "B+Tree").mkdir(parents=True)
    (second / "cuda" / "b-tree").mkdir(parents=True)
    source_registry = tmp_path / "source_registry.json"
    source_registry.write_text(
        json.dumps(
            {
                "schema_version": "source_registry_v1",
                "generated_at": "2026-05-11T00:00:00+00:00",
                "sources": [
                    {
                        "source_id": "source",
                        "local_path": str(first),
                        "availability_status": "source_available",
                    },
                    {
                        "source_id": "source",
                        "local_path": str(second),
                        "availability_status": "source_available",
                    },
                ],
            }
        )
    )

    try:
        build_workload_registry(source_registry, generated_at="2026-05-11T00:00:00+00:00")
    except ValueError as exc:
        assert "source_b-tree" in str(exc)
    else:
        raise AssertionError("Expected duplicate workload_id ValueError")


def test_workload_registry_cli_writes_outputs(tmp_path):
    root = tmp_path / "gpu-parboil"
    (root / "benchmarks" / "bfs" / "src").mkdir(parents=True)
    source_registry = tmp_path / "source_registry.json"
    source_registry.write_text(
        json.dumps(
            {
                "schema_version": "source_registry_v1",
                "generated_at": "2026-05-11T00:00:00+00:00",
                "sources": [
                    {
                        "source_id": "gpu-parboil",
                        "local_path": str(root),
                        "availability_status": "source_available",
                    }
                ],
            }
        )
    )
    output = tmp_path / "registry"

    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "generate_workload_registry.py"),
            "--source-registry",
            str(source_registry),
            "--output-dir",
            str(output),
            "--generated-at",
            "2026-05-11T00:00:00+00:00",
        ],
        check=True,
    )

    registry = json.loads((output / "workload_registry.json").read_text())
    assert registry["generated_at"] == "2026-05-11T00:00:00+00:00"
    assert registry["workloads"][0]["workload_id"] == "gpu-parboil_bfs"
    assert "gpu-parboil_bfs" in (output / "workload_registry.md").read_text()
