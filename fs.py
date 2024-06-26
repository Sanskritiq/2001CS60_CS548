import importlib
import time


def fs(type, feat, label, feat_val, label_val, opts):
    switcher = {
        # 2020
        "mpa": "MarinePredatorsAlgorithm",
        "gndo": "GeneralizedNormalDistributionOptimization",
        "sma": "SlimeMouldAlgorithm",
        "eo": "equilibrium_optimization",
        "mrfo": "MantaRayForagingOptimization",
        # 2019
        'aso': 'atom_search_optimization',
        'hho': 'harris_hawks_optimization',
        'hgso': 'HenryGasSolubilityOptimization',
        'pfa': 'PathFinderAlgorithm',
        'pro': 'PoorAndRichOptimization',
        # 2018
        'boa': 'ButterflyOptimizationAlgorithm',
        'epo': 'emperor_penguin_optimizer',
        'tga': 'TreeGrowthAlgorithm',
        # 2017
        'abo': 'artificial_butterfly_optimization',
        'ssa': 'salp_swarm_algorithm',
        'sbo': 'SatinBowerBirdOptimization',
        'wsa': 'WeightedSuperpositionAttraction',
        # 2016
        'ja': 'jaya_algorithm',
        'csa': 'crow_search_algorithm',
        'sca': 'sine_cosine_algorithm',
        'woa': 'whale_optimization_algorithm',
        # 2015
        'alo': 'ant_lion_optimizer',
        'hlo': 'HumanLearningOptimization',
        'mbo': 'MonarchButterflyOptimization',
        'mfo': 'MothFlameOptimization',
        'mvo': 'MultiVerseOptimizer',
        'tsa': 'TreeSeedAlgorithm',
        # 2014
        'gwo': 'grey_wolf_optimizer',
        'sos': 'SymbioticOrganismsSearch',
        # 2012
        'fpa': 'flower_pollination_algorithm',
        'foa': 'FruitFlyOptimizationAlgorithm',
        # 2009 - 2010
        'ba': 'bat_algorithm',
        'fa': 'firefly_algorithm',
        'cs': 'cuckoo_search_algorithm',
        'gsa': 'gravitational_search_algorithm',
        # Traditional
        'abc': 'artificial_bee_colony',
        'hs': 'harmony_challenge',
        'de': 'differential_evolution',
        'aco': 'ant_colony_optimization',
        'acs': 'ant_colony_system',
        'pso': 'particle_swarm_optimization',
        'gat': 'genetic_algorithm_tour',
        'ga': 'genetic_algorithm',
        'sa': 'SimulatedAnnealing',
    }

    # Get the corresponding module name for the specified type
    module_name = switcher.get(type)
    if module_name is None:
        raise ValueError(f"Invalid feature selection method: {type}")

    # Dynamically import the module
    module = importlib.import_module(module_name)

    start_time = time.time()
    # Call the selected feature selection function
    model = module.optim(feat, label, feat_val, label_val, opts)
    end_time = time.time()
    t = end_time - start_time
    model["t"] = t

    print(f"\n Processing Time (s): {t} \n")

    return model
