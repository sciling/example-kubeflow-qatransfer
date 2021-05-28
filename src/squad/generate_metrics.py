from collections import namedtuple


try:
    from kfp.components import InputPath
except ImportError:

    def InputPath(c):
        return c

Metrics = namedtuple("Outputs", [("mlpipeline_metrics", "Metrics")])


def generate_metrics(
    mlpipelinemetrics_path: InputPath(),
) -> Metrics:
    import json

    with open(mlpipelinemetrics_path, "r") as f:
        metrics = json.load(f)
    print(metrics)
    return [json.dumps(metrics)]
