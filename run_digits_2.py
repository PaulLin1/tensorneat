from tensorneat.pipeline import Pipeline
from tensorneat.algorithm.neat import NEAT
from tensorneat.algorithm.hyperneat import HyperNEAT, FullSubstrate, DefaultSubstrate, MLPSubstrate
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


substrate = MLPSubstrate(
    layers=[65, 100, 50, 10],       # Input: 784 pixels; two hidden layers; output: 10 digits
    coor_range=(-1.0, 1.0, -1.0, 1.0)
)
pipeline = Pipeline(
    algorithm=HyperNEAT(
        substrate=substrate,
        weight_threshold = 0.05,

        neat=NEAT(
            pop_size=100,
            species_size=20,
            survival_threshold=0.4,
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
