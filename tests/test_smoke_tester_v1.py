import os
import tempfile
import pytest
from magda_agent.evaluation.smoke_tester_v1 import SmokeTesterV1


@pytest.fixture
def tester():
    return SmokeTesterV1(verbose=True)


def test_valid_file(tester):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("import sys\nimport os\n\ndef hello():\n    return 'world'\n")
        valid_file = f.name

    try:
        res = tester.evaluate_file(valid_file)
        assert res["valid"] is True
        assert res["syntax"] is True
        assert res["imports"] is True
        assert res["error"] is None

        batch_res = tester.run_smoke_tests([valid_file])
        assert batch_res["success"] is True
        assert batch_res["total_files"] == 1
        assert batch_res["passed_files"] == 1
        assert batch_res["failed_files"] == 0
        assert len(batch_res["errors"]) == 0
    finally:
        if os.path.exists(valid_file):
            os.remove(valid_file)


def test_syntax_error(tester):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("def invalid_syntax(\n    return 42\n")
        invalid_file = f.name

    try:
        res = tester.evaluate_file(invalid_file)
        assert res["valid"] is False
        assert res["syntax"] is False
        assert "SyntaxError" in res["error"]

        batch_res = tester.run_smoke_tests([invalid_file])
        assert batch_res["success"] is False
        assert batch_res["passed_files"] == 0
        assert batch_res["failed_files"] == 1
        assert len(batch_res["errors"]) == 1
    finally:
        if os.path.exists(invalid_file):
            os.remove(invalid_file)


def test_import_error(tester):
    # Testing relative import at top level with level=0 and empty module
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("from . import something\n")
        relative_import_file = f.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("from  import bad_import\n")
        bad_syntax_import_file = f.name

    try:
        # relative import level > 0 is fine, but let's test invalid AST if constructed or check_imports logic
        res = tester.check_imports(bad_syntax_import_file)
        assert res["valid"] is False
    finally:
        if os.path.exists(relative_import_file):
            os.remove(relative_import_file)
        if os.path.exists(bad_syntax_import_file):
            os.remove(bad_syntax_import_file)


def test_missing_file(tester):
    missing_path = "/nonexistent/path/to/script.py"
    res = tester.evaluate_file(missing_path)
    assert res["valid"] is False
    assert "File not found" in res["error"]


def test_mixed_batch(tester):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f1:
        f1.write("x = 10\n")
        file1 = f1.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f2:
        f2.write("class Foo\n    pass\n")
        file2 = f2.name

    try:
        batch_res = tester.run_smoke_tests([file1, file2])
        assert batch_res["success"] is False
        assert batch_res["total_files"] == 2
        assert batch_res["passed_files"] == 1
        assert batch_res["failed_files"] == 1
    finally:
        if os.path.exists(file1):
            os.remove(file1)
        if os.path.exists(file2):
            os.remove(file2)
