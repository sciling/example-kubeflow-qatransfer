import typing

from kfp.components import InputPath


def generate_metrics(
    mlpipelinemetrics_path: InputPath(),
) -> typing.NamedTuple("Outputs", [("mlpipeline_metrics", "Metrics")]):
    import json

    with open(mlpipelinemetrics_path, "r") as f:
        metrics = json.load(f)
    print(metrics)
    return [json.dumps(metrics)]
