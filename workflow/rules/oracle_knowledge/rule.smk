def oracle_knowledge_path():
    return "{output_dir}/oracle_knowledge/{data}/seed={seed}.json"

def oracle_true_graph_from_data(wildcards):
    marker = "adjmat=/"
    if marker not in wildcards.data or "/parameters=" not in wildcards.data:
        raise ValueError("cannot resolve true graph from Benchpress data wildcard")
    adjmat = wildcards.data.split(marker, 1)[1].split("/parameters=", 1)[0]
    return f"{wildcards.output_dir}/adjmat/{adjmat}.csv"


rule oracle_no_trek_sidecar:
    input:
        data=alg_input_data(),
        graph=oracle_true_graph_from_data
    output:
        knowledge=oracle_knowledge_path()
    script:
        "build.py"
