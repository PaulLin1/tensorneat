from tensorneat.pipeline import Pipeline
from tensorneat.algorithm.neat import NEAT
from tensorneat.algorithm.hyperneat import HyperNEAT, FullSubstrate
from tensorneat.genome import DefaultGenome
from tensorneat.common import ACT, AGG
from problems.digits_problem import DigitsClassificationProblem
from tensorneat.genome import DefaultGenome, BiasNode

problem = DigitsClassificationProblem()
# input_coors = [
#     (x / 3.5 - 1.0, y / 3.5 - 1.0)
#     for y in range(8)
#     for x in range(8)
# ]

input_coors = [
    (x / 3.5 - 1.0, y / 3.5 - 1.0)
    for y in range(8)
    for x in range(8)
]
input_coors.append((0.0, -1.2))  # bias node

hidden_coors = [(x / 9 * 2 - 1, y / 9 * 2 - 1) for y in range(10) for x in range(10)]

output_coors = [(x / 9 * 2 - 1, 1) for x in range(10)]

substrate = FullSubstrate(
    input_coors=input_coors,
    hidden_coors=hidden_coors,
    output_coors=output_coors
)
pipeline = Pipeline(
    algorithm=HyperNEAT(
        substrate=substrate,
        neat=NEAT(
            pop_size=100,
            species_size=20,
            survival_threshold=0.2,
            genome=DefaultGenome(
                num_inputs=4,  # size of query coors
                num_outputs=1,
                node_gene=BiasNode(
                    activation_options=[ACT.tanh, ACT.sin, ACT.sigmoid, ACT.gauss, ACT.identity],
                    aggregation_options=[AGG.sum]
                ),
                init_hidden_layers=(8,),
                max_nodes=300,
                max_conns=1000,
                output_transform=ACT.identity
            ),
        ),
        activation=ACT.tanh,
        activate_time=10,
        output_transform=ACT.sigmoid,
    ),
    problem=problem,
    generation_limit=300,
    fitness_target=1,
)

state = pipeline.setup()
state, best = pipeline.auto_run(state)
pipeline.show(state, best)
