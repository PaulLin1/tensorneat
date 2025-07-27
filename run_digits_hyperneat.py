import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
import jax.numpy as jnp
import jax.nn as jnn

from tensorneat.pipeline import Pipeline
from tensorneat.algorithm.neat import NEAT
from tensorneat.algorithm.hyperneat import HyperNEAT, FullSubstrate
from tensorneat.genome import DefaultGenome
from tensorneat.common import ACT
from digits_problem import DigitsClassificationProblem

# 64 inputs from 8x8 grid
input_coors = [
    (x / 3.5 - 1.0, y / 3.5 - 1.0) 
    for y in range(8) 
    for x in range(8)
]

# Add bias coordinate — usually outside normal range, e.g., just below input layer
bias_coor = [(0.0, -1.5)]

input_coors += bias_coor  # now 65 inputs total

# 10 outputs in a line across the top
output_coors = [
    ((i / 4.5 - 1.0), 1.0)
    for i in range(10)
]

# Optional hidden layer (example: 5x2 grid)
hidden_coors = [
    ((x / 2.0 - 1.0), 0.0)
    for x in range(5)
    for y in range(1)
]

substrate = FullSubstrate(
    input_coors=input_coors,
    hidden_coors=hidden_coors,
    output_coors=output_coors,
)

problem = DigitsClassificationProblem()

# the num of input_coors is 5
# 4 is for cartpole inputs, 1 is for bias
pipeline = Pipeline(
    algorithm=HyperNEAT(
        substrate=substrate,
        neat=NEAT(
            pop_size=100,
            species_size=20,
            survival_threshold=0.8,
            genome=DefaultGenome(
                num_inputs=4,  # size of query coors
                num_outputs=1,
                init_hidden_layers=(),
                output_transform=ACT.tanh,
            ),
        ),
        activation=ACT.tanh,
        activate_time=10,
        output_transform = jnn.softmax
    ),
    problem=problem,
    generation_limit=300,
    fitness_target=-1e-6,
)

# initialize state
state = pipeline.setup()
# print(state)
# run until terminate
state, best = pipeline.auto_run(state)