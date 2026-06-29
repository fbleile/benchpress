import time
from pathlib import Path

import numpy as np
import pandas as pd


def wrapper():
    start = time.perf_counter()
    df = pd.read_csv(snakemake.input["data"])
    adjmat = np.zeros((df.shape[1], df.shape[1]), dtype=int)
    pd.DataFrame(adjmat, columns=df.columns).to_csv(snakemake.output["adjmat"], index=False)
    Path(snakemake.output["time"]).write_text(str(time.perf_counter() - start))
    Path(snakemake.output["ntests"]).write_text("0")


wrapper()
