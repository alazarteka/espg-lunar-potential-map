import subprocess
import sys
from pathlib import Path
import pytest

def test_validation_script_demo():
    script_path = Path("scripts/analysis/validate_phys_models.py")
    assert script_path.exists()

    # Run in demo mode
    result = subprocess.run(
        [sys.executable, str(script_path), "--demo"],
        capture_output=True,
        text=True
    )

    assert result.returncode == 0, f"Script failed: {result.stderr}"
    assert "Validation complete" in result.stderr or "Validation complete" in result.stdout
