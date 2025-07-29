from tensorneat.pipeline import Pipeline
from tensorneat.algorithm.neat import NEAT
from tensorneat.algorithm.hyperneat import HyperNEAT, FullSubstrate, DefaultSubstrate, MLPSubstrate
from tensorneat.genome import DefaultGenome
from tensorneat.common import ACT, AGG
from problems.digits_problem import DigitsClassificationProblem
from tensorneat.genome import DefaultGenome, BiasNode
import jax.nn as jnn

from tensorneat.pipeline import Pipeline
from tensorneat.algorithm.neat import NEAT
from tensorneat.algorithm.hyperneat import HyperNEAT, FullSubstrate
from tensorneat.common import ACT, AGG

# 1) True 2D substrate
input_coors = [((x/3.5 - 1.0), (y/3.5 - 1.0))
               for y in range(8) for x in range(8)]
input_coors.append((0.0, -1.2))
output_coors = [(-1.0 + 2.0*i/9.0, 1.2) for i in range(10)]
substrate = FullSubstrate(
    input_coors  = input_coors,
    output_coors = output_coors,
    hidden_layers=[(0.0,0.0)],
)

# 2) CPPN that emits raw weights
hyper = HyperNEAT(
    substrate        = substrate,
    weight_threshold = 1e-3,
    activation       = ACT.tanh,
    output_transform = ACT.identity,
    neat=NEAT(
        pop_size           = 100,
        species_size       = 20,
        survival_threshold = 0.4,
        genome=DefaultGenome(
            num_inputs  = 4,     # CPPN always takes (x1,y1,x2,y2)
            num_outputs = 1,
            node_gene   = BiasNode(
                activation_options=[ACT.tanh, ACT.sin, ACT.gauss, ACT.identity],
                aggregation_options=[AGG.sum]
            ),
            max_nodes  = 300,
            max_conns  = 1000,
        ),
    ),
)

# 3) Pipeline with novelty archive
pipeline = Pipeline(
    algorithm        = hyper,
    problem          = DigitsClassificationProblem(),
    generation_limit = 10000,
    fitness_target   = 1.0,
)

state = pipeline.setup()
state, best = pipeline.auto_run(state)
pipeline.show(state, best)

