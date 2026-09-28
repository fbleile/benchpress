import numpy as np
import subprocess
from pathlib import Path

from udg_pilot.covariance import exact_covbic, fit_covariance, matrix_to_bits, smig_valid


def graph(d, edges):
    a = np.zeros((d, d), dtype=np.uint8)
    for i, j in edges:
        a[i, j] = a[j, i] = 1
    return a


def test_smig_validity_examples():
    assert smig_valid(matrix_to_bits(graph(5, [])), 5)
    assert smig_valid(matrix_to_bits(graph(4, [(i, j) for i in range(4) for j in range(i + 1, 4)])), 4)
    assert smig_valid(matrix_to_bits(graph(4, [(0, 1)])), 4)
    assert smig_valid(matrix_to_bits(graph(4, [(0, 1), (1, 2)])), 4)  # chain
    assert smig_valid(matrix_to_bits(graph(4, [(0, 1), (0, 2)])), 4)  # fork/collider skeleton
    assert smig_valid(matrix_to_bits(graph(6, [(0, 1), (2, 3)])), 6)
    assert not smig_valid(matrix_to_bits(graph(4, [(0, 1), (1, 2), (2, 3), (3, 0)])), 4)


def test_complete_and_empty_covariance_fits():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(500, 3))
    s = (x - x.mean(0)).T @ (x - x.mean(0)) / len(x)
    complete = (1 << 3) - 1  # d=3 has three edge bits
    full = fit_covariance(complete, s, len(x), max_iter=600)
    empty = fit_covariance(0, s, len(x), max_iter=600)
    assert full.converged
    assert np.allclose(full.sigma, s, atol=2e-5)
    assert np.allclose(empty.sigma, np.diag(np.diag(s)), atol=2e-5)
    assert abs(empty.sigma[0, 1]) < 1e-12


def test_two_node_fit_and_bic_penalty():
    rng = np.random.default_rng(5)
    x = rng.multivariate_normal([0, 0], [[1.0, .4], [.4, 2.0]], size=800)
    z = x - x.mean(0)
    s = z.T @ z / len(x)
    fit = fit_covariance(1, s, len(x), max_iter=600)
    assert fit.converged
    assert np.allclose(fit.sigma, s, atol=2e-5)
    diagonal = fit_covariance(0, s, len(x), max_iter=600)
    assert np.isclose(fit.bic - diagonal.bic,
                      (fit.loglik2 - diagonal.loglik2) - np.log(len(x)))


def test_two_node_analytic_delta_matches_rust(tmp_path):
    root = Path(__file__).resolve().parents[2]
    subprocess.run(["cargo", "build", "--release", "--manifest-path",
                    str(root / "udg_pilot/rust/Cargo.toml"), "--bin", "udg_covbic"],
                   cwd=root, check=True, capture_output=True, text=True)
    n = 500; r = 0.2
    s = np.array([[1.0, r], [r, 1.0]])
    path = tmp_path / "s.csv"; path.write_text(",".join(map(str, s.ravel())))
    binary = root / "udg_pilot/rust/target/release/udg_covbic"
    def run(bits):
        text = subprocess.run([str(binary), "score", "2", str(n), str(path), str(bits), "500"],
                              cwd=root, check=True, capture_output=True, text=True).stdout
        out = {}
        for line in text.splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            try:
                out[key] = float(value)
            except ValueError:
                pass
        return out
    delta = run(1)["loglik2"] - run(0)["loglik2"]
    expected = -n * np.log(1 - r*r)
    assert np.isclose(delta, expected, rtol=1e-5, atol=1e-5)
    assert np.isclose(run(1)["bic"] - run(0)["bic"], expected - np.log(n),
                      rtol=1e-5, atol=1e-5)


def test_exact_tiny_enumeration_is_available():
    rng = np.random.default_rng(6)
    x = rng.normal(size=(80, 3)); z = x - x.mean(0)
    s = z.T @ z / len(x)
    best = exact_covbic(s, len(x), valid_only=True)
    assert best is not None
    assert smig_valid(best[0], 3)
