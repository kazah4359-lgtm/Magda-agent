import os
import tempfile
import pytest
from magda_agent.evaluation.smoke_tester_v2 import SmokeTesterV2

@pytest.fixture
def tester():
    return SmokeTesterV2()

def test_syntax_errors(tester):
    # Valid syntax
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("def foo():\n    return 42\n")
        valid_file = f.name

    # Invalid syntax
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("def foo() -> int\n    return 'not an int'\n")
        invalid_file = f.name

    try:
        assert tester.check_syntax(valid_file) is True
        assert tester.check_syntax(invalid_file) is False
    finally:
        os.remove(valid_file)
        os.remove(invalid_file)

def test_import_errors(tester):
    # Assuming well-formed import
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("import os\nfrom typing import List\n")
        valid_file = f.name

    try:
        assert tester.check_imports(valid_file) is True
    finally:
        os.remove(valid_file)

def test_valid_file(tester):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("import sys\n\ndef main():\n    pass\n")
        valid_file = f.name

    try:
        result = tester.run_all_checks([valid_file])
        assert result["success"] is True
        assert valid_file in result["details"]
        assert result["details"][valid_file]["valid"] is True
    finally:
        os.remove(valid_file)
