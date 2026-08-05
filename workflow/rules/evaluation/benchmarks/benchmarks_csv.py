import os
import json

def dict_to_summary(d):
    s = ""
    for key, val in d.items():
        # Quote values because DAGMA's continuation schedule contains commas;
        # without shell quoting those values can produce malformed CSV rows.
        s += "python workflow/rules/evaluation/benchmarks/add_column.py --filename {res} --colname "+key+" --colval '{"+key+"}' \n "
    return s

algorithm = snakemake.params["alg"]

with open(snakemake.params["config"]) as json_file:    
    config = json.load(json_file)

# Run the legacy shell chain fail-fast.  Without ``set -e`` a failed R
# summarizer is followed by add_column.py, which hides the real error behind
# a misleading FileNotFoundError for result.csv.
cmd="set -e\n"
cmd += """
Rscript workflow/rules/evaluation/benchmarks/run_summarise.R  --adjmat_true {adjmat_true} --adjmat_est {adjmat_est}  --filename {res}  
python workflow/rules/evaluation/benchmarks/add_column.py --filename {res} --colname seed       --colval {seed}
python workflow/rules/evaluation/benchmarks/add_column.py --filename {res} --colname algorithm       --colval {alg} 
python workflow/rules/evaluation/benchmarks/add_column.py --filename {res} --colname adjmat          --colval {adjmat} 
python workflow/rules/evaluation/benchmarks/add_column.py --filename {res} --colname parameters      --colval {bn} 
python workflow/rules/evaluation/benchmarks/add_column.py --filename {res} --colname data            --colval {data} 
python workflow/rules/evaluation/benchmarks/add_column.py --filename {res} --colname time            --colval `cat {time}`
python workflow/rules/evaluation/benchmarks/add_column.py --filename {res} --colname ntests          --colval None 
"""
algorithm_config = config["resources"]["structure_learning_algorithms"][algorithm][0]
cmd += dict_to_summary(algorithm_config)

# Compact DAGMA path identities intentionally omit schema-default wildcards.
# The benchmark summary still records the complete effective configuration;
# fill omitted path fields from the validated config before formatting.
format_values = {
    **dict(snakemake.input), **dict(snakemake.params),
    **dict(snakemake.output), **dict(snakemake.wildcards),
}
for key, value in algorithm_config.items():
    format_values.setdefault(key, value)
# Store list-like configuration values as a single metadata field.  In
# particular, ``s`` is a comma-separated continuation schedule; leaving it
# unquoted in the legacy shell-based add-column chain creates ragged CSV rows
# on some cluster shells.
for key in algorithm_config:
    value = format_values.get(key)
    if isinstance(value, (list, tuple)):
        format_values[key] = ";".join(str(item) for item in value)
    elif isinstance(value, str) and "," in value:
        format_values[key] = value.replace(",", ";")
command = cmd.format(**format_values)

os.system(command)
