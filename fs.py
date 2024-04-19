import importlib
import time


def fs(type, feat, label, feat_val, label_val, opts):
    switcher = {
        # 2020
        "mpa": "marine_predators_algorithm",
        "gndo": "generalized_nondominated_sorting",
        "sma": "slime_mould_algorithm",
        "eo": "equilibrium_optimization",
        "mrfo": "manta_ray_foraging_optimization",
        # 2019
        "aso": "jAtomSearchOptimization",
        "hho": "jHarrisHawksOptimization",
        "hgso": "jHenryGasSolubilityOptimization",
        "pfa": "jPathFinderAlgorithm",
        "pro": "jPoorAndRichOptimization",
        # 2018
        "boa": "jButterflyOptimizationAlgorithm",
        "epo": "jEmperorPenguinOptimizer",
        "tga": "jTreeGrowthAlgorithm",
        # 2017
        "abo": "jArtificialButterflyOptimization",
        "ssa": "jSalpSwarmAlgorithm",
        "sbo": "jSatinBowerBirdOptimization",
        "wsa": "WeightedSuperpositionAttraction",
        # 2016
        "ja": "jJayaAlgorithm",
        "csa": "jCrowSearchAlgorithm",
        "sca": "jSineCosineAlgorithm",
        "woa": "jWhaleOptimizationAlgorithm",
        # 2015
        "alo": "jAntLionOptimizer",
        "hlo": "jHumanLearningOptimization",
        "mbo": "jMonarchButterflyOptimization",
        "mfo": "jMothFlameOptimization",
        "mvo": "jMultiVerseOptimizer",
        "tsa": "TreeSeedAlgorithm",
        # 2014
        "gwo": "jGreyWolfOptimizer",
        "sos": "jSymbioticOrganismsSearch",
        # 2012
        "fpa": "jFlowerPollinationAlgorithm",
        "foa": "jFruitFlyOptimizationAlgorithm",
        # 2009 - 2010
        "ba": "jBatAlgorithm",
        "fa": "jFireflyAlgorithm",
        "cs": "jCuckooSearchAlgorithm",
        "gsa": "jGravitationalSearchAlgorithm",
        # Traditional
        "abc": "artificial_bee_colony",
        "hs": "jHarmonySearch",
        "de": "jDifferentialEvolution",
        "aco": "jAntColonyOptimization",
        "acs": "jAntColonySystem",
        "pso": "jParticleSwarmOptimization",
        "gat": "jGeneticAlgorithmTour",
        "ga": "jGeneticAlgorithm",
        "sa": "jSimulatedAnnealing",
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
