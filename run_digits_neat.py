import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from tensorneat.pipeline import Pipeline
from tensorneat.algorithm.neat.neat import NEAT  # <- add this
from digits_problem import DigitsClassificationProblem


import jax.numpy as jnp

from tensorneat.pipeline import Pipeline
from tensorneat.algorithm.neat import NEAT
from tensorneat.genome import DefaultGenome, BiasNode
from tensorneat.problem.func_fit import CustomFuncFit
from tensorneat.common import ACT, AGG

problem = DigitsClassificationProblem()

pipeline = Pipeline(
    algorithm=NEAT(
        pop_size=100,
        species_size=20,
        survival_threshold=0.01,
        genome=DefaultGenome(
            num_inputs=64,
            num_outputs=10,
            init_hidden_layers=(),
            node_gene=BiasNode(
                activation_options=[ACT.identity, ACT.inv],
                aggregation_options=[AGG.sum, AGG.product],
            ),
            max_nodes=1000,
            max_conns=1000,
            output_transform=ACT.identity,
        ),
    ),
    problem=problem,
    generation_limit=100,
    fitness_target=-1e-4,
    seed=42,
)

# initialize state
state = pipeline.setup()
# run until terminate
state, best = pipeline.auto_run(state)
# show result
pipeline.show(state, best)
