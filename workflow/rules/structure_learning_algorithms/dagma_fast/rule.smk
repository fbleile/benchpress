rule:
    name: module_name
    input:
        data=alg_input_data()
    output:
        adjmat=alg_output_adjmat_path(module_name),
        time=alg_output_time_path(module_name),
        ntests=alg_output_ntests_path(module_name),
        diagnostics="{output_dir}/diagnostics/{data}/algorithm=/" + pattern_strings[module_name] + "/seed={seed}/diagnostics.json"
    container:
        "docker://bpimages/dagma:1.1.1"
    script:
        "run.py"
