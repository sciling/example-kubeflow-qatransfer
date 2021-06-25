try:
    from kfp.components import InputPath
    from kfp.components import OutputPath
except ImportError:

    def InputPath(c):
        return c

    def OutputPath(c):
        return c


metrics = "Metrics"


def semeval_test(
    test_path: InputPath(str),
    run_id,
    start_step,
    end_step,
    th,
    reranking_th,
    op_format,
    verbose,
    ignore_noanswer,
    mlpipeline_metrics_path: OutputPath(metrics),
):
    from semeval.evaluation.MAP_scripts.ev import eval_reranker

    th = int(th)
    reranking_th = int(reranking_th)

    def main(options, args):
        res_fname = args[0]
        pred_fname = args[1]
        return eval_reranker(
            res_fname,
            pred_fname,
            options["format"],
            options["th"],
            options["verbose"],
            options["reranking_th"],
            options["ignore_noanswer"],
        )

    start_step = int(start_step)
    end_step = int(end_step)
    if verbose == "False":
        verbose = False
    else:
        verbose = True
    if ignore_noanswer == "False":
        ignore_noanswer = False
    else:
        ignore_noanswer = True
    options = {
        "th": th,
        "reranking_th": reranking_th,
        "format": op_format,
        "verbose": verbose,
        "ignore_noanswer": ignore_noanswer,
    }
    metric_list = []
    for step in range(start_step, end_step + 1, 200):
        args = [
            test_path + "/semeval/store/test-gold",
            test_path + "/semeval/store/test-" + run_id + "-" + str(step).zfill(6),
        ]
        map_svm, mrr_svm, avg_acc1_svm = main(options, args)
        metric_list.append(
            {
                "name": "MAP-test-" + run_id + "-" + str(step).zfill(6),
                "numberValue": str(map_svm),
                "format": "RAW",
            }
        )
        metric_list.append(
            {
                "name": "MRR-test-" + run_id + "-" + str(step).zfill(6),
                "numberValue": str(mrr_svm),
                "format": "RAW",
            }
        )
        metric_list.append(
            {
                "name": "AvgRec-test-" + run_id + "-" + str(step).zfill(6),
                "numberValue": str(avg_acc1_svm),
                "format": "RAW",
            }
        )
    metrics = {"metrics": metric_list}

    import json

    with open(mlpipeline_metrics_path, "w") as f:
        json.dump(metrics, f)
