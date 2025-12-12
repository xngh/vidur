import json
from typing import Dict

from vidur.config import (
    BaseRequestGeneratorConfig,
    ClusterConfig,
    HeterClusterConfig,
    MetricsConfig,
)
from vidur.entities.base_entity import BaseEntity
from vidur.entities.replica import Replica
from vidur.logger import init_logger

logger = init_logger(__name__)


class Cluster(BaseEntity):
    def __init__(
        self,
        cluster_config: ClusterConfig,
        metrics_config: MetricsConfig,
        generator_config: BaseRequestGeneratorConfig,
    ) -> None:
        self._id = Cluster.generate_id()
        self._config = cluster_config

        # get metrics config
        self._output_dir = metrics_config.output_dir

        # Init replica object handles
        self._replicas: Dict[str, Replica] = {}

        for _ in range(self._config.num_replicas):
            replica = Replica(self._config.replica_config, generator_config)
            self._replicas[replica.id] = replica

        if metrics_config.write_json_trace:
            self._write_cluster_info_to_file()

    @property
    def replicas(self):
        return self._replicas

    def to_dict(self) -> dict:
        return {
            "id": self._id,
            "num_replicas": len(self._replicas),
        }

    @classmethod
    def from_heter_config(
        cls,
        cluster_config: HeterClusterConfig,
        metrics_config: MetricsConfig,
        generator_config: BaseRequestGeneratorConfig,
    ):
        """
        异构集群构造函数：逐个 ReplicaConfig 实例化 Replica。
        保持 __init__ 同构语义，避免混淆。
        """
        self = cls.__new__(cls)  # bypass __init__
        self._id = Cluster.generate_id()
        self._config = cluster_config
        self._output_dir = metrics_config.output_dir
        self._replicas: Dict[str, Replica] = {}

        for replica_config in cluster_config.replica_configs:
            replica = Replica(replica_config, generator_config)
            self._replicas[replica.id] = replica

        if metrics_config.write_json_trace:
            self._write_cluster_info_to_file()

        return self

    def _write_cluster_info_to_file(self) -> None:
        replica_dicts = [replica.to_dict() for replica in self._replicas.values()]
        cluster_info = {"replicas": replica_dicts}

        cluster_file = f"{self._output_dir}/cluster.json"
        with open(cluster_file, "w") as f:
            json.dump(cluster_info, f)


if __name__ == "__main__":
    """
    简单单测：使用 config.py 中的 SimulationConfig.create_from_cli_args_heter_full
    构造异构集群配置，并用 Cluster.from_heter_config 构建 Cluster，打印结构。
    运行方式（在项目根目录）:
        PYTHONPATH=. python -m vidur.entities.cluster \\
          --cluster_config_replica_configs '[{"device": "h100", "model_name": "Qwen/Qwen-72B", "count": 1}, {"device": "a100", "model_name": "meta-llama/Llama-2-7b-hf", "count": 3}]'
    """
    import sys
    import json as pyjson
    from vidur.config.config import SimulationConfig, HeterClusterConfig

    # 如果命令行未提供，则使用默认示例
    if "--cluster_config_replica_configs" not in sys.argv:
        sys.argv = [
            sys.argv[0],
            "--cluster_config_replica_configs",
            '[{"device": "a100", "model_name": "Qwen/Qwen-72B", "count": 1}, {"device": "a100", "model_name": "meta-llama/Llama-2-7b-hf", "count": 1}]',
        ]

    config = SimulationConfig.create_from_cli_args_heter_full()
    # config = SimulationConfig.create_from_cli_args()

    if isinstance(config.cluster_config, HeterClusterConfig):
        cluster = Cluster.from_heter_config(
            config.cluster_config, config.metrics_config, config.request_generator_config
        )
    else:
        cluster = Cluster(config.cluster_config, config.metrics_config, config.request_generator_config)

    print("Cluster summary (dict):")
    print(pyjson.dumps(cluster.to_dict(), indent=2, ensure_ascii=False))

    # 详细打印各 replica 配置
    replicas_detail = [replica.to_dict() for replica in cluster.replicas.values()]
    print("\nReplicas detail:")
    print(pyjson.dumps(replicas_detail, indent=2, ensure_ascii=False))
