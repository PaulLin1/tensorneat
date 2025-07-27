import sys
import os
from problems.mnist_problem import MNISTClassificationProblem  # ← your custom problem file

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from tensorneat.pipeline import Pipeline
from tensorneat.algorithm.neat import NEAT
from tensorneat.genome import DefaultGenome, BiasNode
from tensorneat.common import ACT, AGG


import jax.numpy as jnp

# Use MNIST instead of Digits
problem = MNISTClassificationProblem()

pipeline = Pipeline(
    algorithm=NEAT(
        pop_size=500,
        species_size=20,
        survival_threshold=0.01,
        genome=DefaultGenome(
            num_inputs=784,  # ← MNIST input (28x28)
            num_outputs=10,
            init_hidden_layers=(),
            node_gene=BiasNode(
                activation_options=[ACT.identity, ACT.inv],
                aggregation_options=[AGG.sum, AGG.product],
            ),
            max_nodes=2000,
            max_conns=10000,
            output_transform=ACT.identity,
        ),
    ),
    problem=problem,
    generation_limit=500,
    fitness_target=1.0,  # ← 100% accuracy
    seed=42,
)

# initialize state
state = pipeline.setup()
# run until terminate
state, best = pipeline.auto_run(state)
# show result
pipeline.show(state, best)
