import sys
import os

from tensorneat.pipeline import Pipeline
from tensorneat.algorithm.neat.neat import NEAT  # <- add this

import jax.numpy as jnp

from tensorneat.pipeline import Pipeline
from tensorneat.algorithm.neat import NEAT
from tensorneat.genome import DefaultGenome, BiasNode
from tensorneat.problem.func_fit import CustomFuncFit
from tensorneat.common import ACT, AGG
from problems.digits_problem import DigitsClassificationProblem

problem = DigitsClassificationProblem()

pipeline = Pipeline(
    algorithm=NEAT(
        pop_size=500,
        species_size=20,
        survival_threshold=0.01,
        genome=DefaultGenome(
            num_inputs=64,
            num_outputs=10,
            init_hidden_layers=(),
            # node_gene=BiasNode(
            #     activation_options=[ACT.identity, ACT.inv],
            #     aggregation_options=[AGG.sum, AGG.product],
            # ),
            max_nodes=2000,
            max_conns=5000,
            output_transform=ACT.identity,
        ),
    ),
    problem=problem,
    generation_limit=500,
    fitness_target=.9,
    seed=42,
)

# initialize state
state = pipeline.setup()
# run until terminate
state, best = pipeline.auto_run(state)
# show result
pipeline.show(state, best)
