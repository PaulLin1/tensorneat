import sys
import os

from tensorneat.pipeline import Pipeline
from tensorneat.algorithm.neat.neat import NEAT

import jax.numpy as jnp

from tensorneat.pipeline import Pipeline
from tensorneat.algorithm.neat import NEAT
from tensorneat.genome import DefaultGenome, BiasNode
from tensorneat.algorithm.hyperneat import HyperNEAT, FullSubstrate
from tensorneat.common import ACT, AGG
from problems.mnist_problem import MNISTClassificationProblem

problem = MNISTClassificationProblem()


from tensorneat.algorithm.hyperneat import FullSubstrate

# 28x28 input coordinates normalized to [-1, 1]
input_coors = [
    (x / 13.5 - 1.0, y / 13.5 - 1.0)
    for y in range(28)
    for x in range(28)
]

# Add bias unit below the grid
input_coors.append((0.0, -1.2))

# Optional: wider hidden layer
hidden_coors = [((x / 4.0 - 1.0), 0.0) for x in range(9)]  # 9 nodes

# Output layer for 10 classes
output_coors = [((x / 4.5 - 1.0), 1.0) for x in range(10)]

substrate = FullSubstrate(
    input_coors=input_coors,
    hidden_coors=hidden_coors,
    output_coors=output_coors,
)

pipeline = Pipeline(
    algorithm=HyperNEAT(
        substrate=substrate,
        neat=NEAT(
            pop_size=500,
            species_size=20,
            survival_threshold=0.01,
            genome=DefaultGenome(
                num_inputs=4,  # size of query coors
                num_outputs=1,
                init_hidden_layers=(),
                output_transform=ACT.tanh,
            ),
        ),
        activation=ACT.tanh,
        activate_time=10,
        output_transform=jnp.argmax,
    ),
    problem=problem,
    generation_limit=500,
    fitness_target=.8,
    seed=42,
)

# initialize state
state = pipeline.setup()
# run until terminate
state, best = pipeline.auto_run(state)
# show result
pipeline.show(state, best)
