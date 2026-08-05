import time

try:
    import flopsearch
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "FLOP requires the Linux flopsearch==0.3.0 package in the active "
        "Python environment. In host mode run "
        "bash scripts/install_flopsearch_host.sh before starting the farm."
    ) from exc
import numpy as np
import pandas as pd
from workflow.rules.structure_learning_algorithms.flop.adapter import convert_flop_cpdag

df = pd.read_csv(snakemake.input["data"])
X = df.to_numpy(dtype=float)
means, stds = X.mean(0), X.std(0, ddof=0)
if np.any(stds == 0):
    raise ValueError("FLOP cannot standardize a constant column")
X = (X - means) / stds
kwargs = {}
restarts = snakemake.wildcards["restarts"]
timeout = snakemake.wildcards["search_timeout"]
if str(restarts) not in {"None", "null"}:
    kwargs["restarts"] = int(restarts)
elif str(timeout) not in {"None", "null"}:
    kwargs["timeout"] = float(timeout)
else:
    raise ValueError("FLOP requires restarts or search_timeout")
start = time.perf_counter()
raw = np.asarray(flopsearch.flop(X, float(snakemake.wildcards["lambda_bic"]), **kwargs))
elapsed = time.perf_counter() - start
A = convert_flop_cpdag(raw, X.shape[1])
pd.DataFrame(A.astype(int), columns=df.columns).to_csv(snakemake.output["adjmat"], index=False)
with open(snakemake.output["time"], "w") as handle:
    handle.write(str(elapsed))
with open(snakemake.output["ntests"], "w") as handle:
    handle.write("None")
