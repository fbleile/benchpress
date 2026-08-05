def oracle_knowledge_path():
    return ("{output_dir}/oracle_knowledge/adjmat=/{adjmat}/"
            "parameters=/{bn}/data=/{data}/seed={seed}.json")

def oracle_true_graph_from_data(wildcards):
    return f"{wildcards.output_dir}/adjmat/{wildcards.adjmat}.csv"


rule oracle_no_trek_sidecar:
    input:
        data=alg_input_data(),
        graph=oracle_true_graph_from_data
    output:
        knowledge=oracle_knowledge_path()
    script:
        "build.py"
