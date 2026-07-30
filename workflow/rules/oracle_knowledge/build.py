import pandas as pd

from workflow.rules.structure_learning_algorithms.dagma.knowledge import write_oracle_sidecar

data = pd.read_csv(snakemake.input["data"])
graph = pd.read_csv(snakemake.input["graph"]).to_numpy()
write_oracle_sidecar(graph, list(data.columns), snakemake.output["knowledge"])
