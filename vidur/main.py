from vidur.agent_simulator import AgentSimulator
from vidur.config import SimulationConfig
from vidur.simulator import Simulator
from vidur.utils.random import set_seeds


def main() -> None:
    config: SimulationConfig = SimulationConfig.create_from_cli_args()

    set_seeds(config.seed)

    simulator = Simulator(config)
    simulator.run()

def mainv2() -> None:
    config: SimulationConfig = SimulationConfig.create_from_cli_args()

    set_seeds(config.seed)

    simulator = AgentSimulator(config)
    simulator.run()

if __name__ == "__main__":
    mainv2()
