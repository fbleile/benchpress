# This file essentially defines the pattern strings for all algorithms, 
# 
# 
# TODO: It would be if good one could have different patterns for the same algorithm, 
# so that one could omit some paramters e.g.
#

# The pattern strings are generated from the json config file.
pattern_strings = {}

_DAGMA_PATH_DEFAULTS = {
    "variance_epsilon": 1e-8,
    "selection_mode": "continuous_objective",
    "bic_constraint_mode": "require_zero_violations",
    "standardize_data": True,
    "lambda_policy": "fixed_0.03",
    "regularizer_type": "L1",
    "postselection_policy": "PS1_joint_feasible_greedy_score",
    "candidate_edge_pool": "threshold_grid",
    "threshold_grid": [0.01, 0.03, 0.05, 0.10, 0.20, 0.30],
    "fixed_threshold": 0.30,
    "max_search_seconds": 1.0,
    "max_expanded_nodes": 1000,
    "max_queue_size": 1000,
    "max_ambiguous_edges": 20,
    "max_indegree": None,
    "dag_constraint_active": True,
    "notreks_constraint_active": True,
    "constraint_regime": None,
    "lambda1_scaling": "fixed",
    "lambda1_reference": 0.03,
    "lambda1_reference_d": 50,
    "lambda1_reference_n": 1000,
    "mu_schedule": None,
    "terminal_zero_stage": False,
    "zero_block_iterations": 10000,
    "maximum_zero_iterations": 300000,
    "h_tolerance": 1e-12,
    "notreks_tolerance": 1e-12,
    "feasibility_threshold_tolerance": 1e-6,
    "gradient_tolerance": 1e-8,
    "dag_penalty_weight": 1.0,
    "dag_constraint": "logdet",
    "gamma_inv": 1.0,
    "restarts": 1,
    "n_jobs": 1,
    "algorithm_seed": 0,
    "initialization_scale": 0.05,
    "trek_function": "inv",
    "trek_kernel": "fast",
    "knowledge_fraction": 1.0,
    "knowledge_seed": 0,
}


def _compact_algorithm_config(alg, values):
    """Drop schema-default DAGMA fields from path identities.

    Schema validation can add many optional defaults.  Encoding every one in
    the output directory makes otherwise valid smoke paths exceed filesystem
    component limits.  Non-default values remain in the identity, while the
    run scripts use the same defaults when a wildcard is absent.
    """
    if alg not in {"dagma", "dagma_notreks"}:
        return values
    if isinstance(values, list):
        return [{k: v for k, v in item.items()
                 if k not in _DAGMA_PATH_DEFAULTS or v != _DAGMA_PATH_DEFAULTS[k]}
                for item in values]
    defaults = _DAGMA_PATH_DEFAULTS
    return {k: v for k, v in values.items()
            if k not in defaults or v != defaults[k]}


def get_algorithm_patterns(config):
    """Generate pattern strings for structure learning algorithms"""
    patterns = {}
    for alg in config["resources"]["structure_learning_algorithms"].keys():
        values = _compact_algorithm_config(
            alg, config["resources"]["structure_learning_algorithms"][alg])
        patterns[alg] = alg+"/alg_params=/"+dict_to_path(values)
    return patterns

def get_graph_patterns(config):
    """Generate pattern strings for graph modules"""
    patterns = {}
    for module in config["resources"]["graph"]:
        patterns[module] = module + "/" + dict_to_path(config["resources"]["graph"][module])
    return patterns

def get_parameter_patterns(config):
    """Generate pattern strings for parameter modules"""
    patterns = {}
    for module in config["resources"]["parameters"]:
        patterns[module] = module + "/" + dict_to_path(config["resources"]["parameters"][module])
    return patterns

def get_data_patterns(config):
    """Generate pattern strings for data modules"""
    patterns = {}
    for module in config["resources"]["data"]:
        patterns[module] = module + "/" + dict_to_path(config["resources"]["data"][module])
    return patterns

def get_mcmc_eval_patterns(config):
    """Generate pattern strings for MCMC evaluation methods"""
    patterns = {}
    for mcmc_eval in ["mcmc_traj_plots", "mcmc_autocorr_plots", "mcmc_heatmaps"]:
        for bmark_setup in config["benchmark_setup"]:
            if mcmc_eval in bmark_setup["evaluation"]:
                patterns[mcmc_eval] = mcmc_eval + "/" + dict_to_path(bmark_setup["evaluation"][mcmc_eval])
    return patterns

def get_mcmc_est_pattern():
    """Generate pattern string for MCMC estimation parameters"""
    return "mcmc_params/"\
           "mcmc_estimator={mcmc_estimator}/"\
           "threshold={threshold}/"\
           "burnin_frac={burnin_frac}"

# Initialize empty pattern strings dict
pattern_strings = {}

# Build up pattern strings by calling each function
# May be good to add all since they might be input for other algs.
pattern_strings.update(get_algorithm_patterns(config))
pattern_strings.update(get_graph_patterns(config))
pattern_strings.update(get_parameter_patterns(config))
pattern_strings.update(get_data_patterns(config))
pattern_strings.update(get_mcmc_eval_patterns(config))
pattern_strings["mcmc_est"] = get_mcmc_est_pattern()
