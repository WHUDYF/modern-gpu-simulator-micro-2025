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


def init_git_repo(path: Path) -> str:
    path.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=path, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=path, check=True)
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
