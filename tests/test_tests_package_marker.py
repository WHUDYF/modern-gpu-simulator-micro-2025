from pathlib import Path

import tests


def test_tests_package_resolves_to_repository_tests_dir():
    assert Path(tests.__file__).resolve().parent == Path(__file__).resolve().parent
