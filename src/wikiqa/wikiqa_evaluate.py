from kfp.components import InputPath
from kfp.components import OutputPath


metrics = "Metrics"


def wikiqa_evaluate(
    wikiqa_path: InputPath(str),
    model_path: InputPath(str),
    start_step,
    end_step,
    eval_period,
    run_ids,
    mlpipeline_metrics_path: OutputPath(metrics),
):
    from wikiqa.result import evaluate
    from wikiqa.result import load

    end_step = int(end_step)
    start_step = int(start_step)
    eval_period = int(eval_period)

    def get_args():
        from types import SimpleNamespace

        data_dir = wikiqa_path + "/wikiqa-class"
        eval_dir = model_path + "/out/wikiqa/basic-class"
        args = SimpleNamespace(
            data_dir=data_dir,
            eval_dir=eval_dir,
            run_ids=run_ids,
            eval_name="test",
            eval_period=eval_period,
            start_step=start_step,
            end_step=end_step,
            steps="",
            ensemble=False,
        )
        return args

    def main():
        metrics_list = []
        args = get_args()
        data = load(args)
        for run_id in args.run_ids.split(","):
            best_eval, best_global_step = (0, 0, 0), -1
            print("Evaluate run_id = %s..." % run_id)
            for global_step in range(
                args.start_step, args.end_step + args.eval_period, args.eval_period
            ):
                curr_eval = evaluate(args, [run_id], data, [global_step])
                if curr_eval[0] > best_eval[0]:
                    best_eval, best_global_step = curr_eval, global_step
            print(
                "Best MAP: %.2f\tMRR: %.2f\tP@1: %.2f in global step %d"
                % (best_eval[0], best_eval[1], best_eval[2], best_global_step)
            )

            """Generating metrics for the squad model"""
            metrics_list.append(
                {
                    "name": "MAP_for_run_%d" % best_global_step,
                    "numberValue": str(best_eval[0]),
                    "format": "RAW",
                }
            )
            metrics_list.append(
                {
                    "name": "MRR_for_run_%d" % best_global_step,
                    "numberValue": str(best_eval[1]),
                    "format": "RAW",
                }
            )
            metrics_list.append(
                {
                    "name": "P1_for_run_%d" % best_global_step,
                    "numberValue": str(best_eval[2]),
                    "format": "RAW",
                }
            )

        metrics = {"metrics": metrics_list}

        import json

        with open(mlpipeline_metrics_path, "w") as f:
            json.dump(metrics, f)

    main()
