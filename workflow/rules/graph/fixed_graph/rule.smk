
rule fixed_adjmat:
    wildcard_constraints:
        output_dir="results"
    input:
        "resources/adjmat/myadjmats/{adjmat}.csv"
    output:
        "{output_dir}/adjmat/myadjmats/{adjmat}.csv" 
    shell:        
        "cp {input} {output}"
