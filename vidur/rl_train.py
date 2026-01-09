from vidur.agent_simulator import AgentSimulator
from vidur.config import SimulationConfig
from vidur.simulator import Simulator
from vidur.utils.random import set_seeds


def main() -> None:
    config: SimulationConfig = SimulationConfig.create_from_cli_args()

    set_seeds(config.seed)

    simulator = AgentSimulator(config)


    for i in range(10):
        simulator.reset()
        simulator.run()
        print(f"Train Iteration: {i} finished.")

if __name__ == "__main__":
    main()
