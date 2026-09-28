import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "udg_pilot" / "rust" / "Cargo.toml"


def test_rust_source_mask_builds_and_is_reproducible(tmp_path):
    subprocess.run(["cargo", "build", "--release", "--manifest-path", str(MANIFEST)],
                   cwd=ROOT, check=True, capture_output=True, text=True)
    binary = ROOT / "udg_pilot" / "rust" / "target" / "release" / "udg_source_mask"
    evidence = tmp_path / "evidence.csv"
    evidence.write_text("0,1,2,0,0,3,4,0,0,0,0,1,2,0,3,4")
    command = [str(binary), "4", str(evidence), "17", "3", "100", "greedy", "1.0"]
    first = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True).stdout
    second = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True).stdout
    assert first == second
    assert "masks=" in first and "incremental_evaluations=" in first

