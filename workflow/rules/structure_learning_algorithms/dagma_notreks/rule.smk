def dagma_notreks_knowledge(wildcards):
    source = str(wildcards.knowledge_source)
    if source == "oracle_true_graph":
        return oracle_knowledge_path()
    if source == "file":
        return str(wildcards.knowledge_file)
    if source == "none":
        return []
    raise ValueError("knowledge_source must be oracle_true_graph, file, or none")


rule:
    name: module_name
    input:
        data=alg_input_data(),
        knowledge=dagma_notreks_knowledge
    output:
        adjmat=alg_output_adjmat_path(module_name),
        time=alg_output_time_path(module_name),
        ntests=alg_output_ntests_path(module_name),
        diagnostics=alg_output_adjmat_path(module_name) + ".diagnostics.json"
    container:
        "docker://bpimages/dagma:1.1.1"
    script:
        "run.py"
