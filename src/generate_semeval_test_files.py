from kfp.components import OutputPath, InputPath


def generate_semeval_test_files(semeval_path : InputPath(str), model_path : InputPath(str), start_step, end_step,
                                eval_period, run_ids, threshold, test_path : OutputPath(str)):
    import os
    from semeval.result import load_data, evaluate

    end_step = int(end_step)
    start_step = int(start_step)
    eval_period = int(eval_period)
    threshold = float(threshold)

    def get_args():
        data_dir = semeval_path + '/semeval'
        eval_dir = model_path + '/out/semeval/basic-class'
        store_dir = test_path + '/semeval/store'
        from types import SimpleNamespace
        args = SimpleNamespace(data_dir=data_dir, end_step=end_step, ensemble=False,
                               eval_dir=eval_dir, eval_name='test', eval_period=eval_period,
                               run_ids=run_ids, start_step=start_step, steps='',
                               store_dir=store_dir, threshold=threshold)
        return args

    def main():
        args = get_args()
        print(args)
        if args.ensemble:
            print("Ensemble not implemented yet")
            return

        data = load_data(args)
        if not os.path.exists(args.store_dir):
            os.makedirs(args.store_dir)

        for i, run_id in enumerate(args.run_ids.split(',')):
            for step in range(args.start_step, args.end_step, args.eval_period):
                evaluate(args, data, run_id, step, dump_gold=(i == 0 and step == args.start_step))

    main()
    print('Generated files:')
    print(os.listdir(test_path + '/semeval/store'))
