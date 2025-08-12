import sys
import os
from tensorneat.pipeline import Pipeline
from tensorneat.algorithm.neat import NEAT
from tensorneat.genome import DefaultGenome, BiasNode
from tensorneat.common import ACT, AGG
from problems.mnist_problem import MNISTClassificationProblem  # ← your custom problem file
import jax.numpy as jnp
# Use MNIST instead of Digits


problem = MNISTClassificationProblem()
algorithm=NEAT(
    pop_size=500,
    species_size=75,
    survival_threshold=0.2,
    compatibility_threshold=.905,
    genome=DefaultGenome(
        num_inputs=784,  # ← MNIST input (28x28)
        num_outputs=10,
        max_nodes=900,
        max_conns=20000,
    ),
)

pipeline = Pipeline(
    algorithm=algorithm,
    problem=problem,
    generation_limit=500,
    fitness_target=1,
    seed=42,
    is_save=True, save_dir='models/'
)

# initialize state
state = pipeline.setup()
# run until terminate
state, best = pipeline.auto_run(state)
# show result
pipeline.show(state, best)


