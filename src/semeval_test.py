from kfp.components import OutputPath, InputPath


def step5(test_path: InputPath(str), run_id, start_step, end_step, th, reranking_th, op_format, verbose, ignore_noanswer):
    import sys
    from semeval.evaluation.MAP_scripts.ev import eval_search_engine, eval_reranker

    th = int(th)
    reranking_th = int(reranking_th)

    def main(options, args):
        if len(args) == 1:
            res_fname = args[0]
            eval_search_engine(res_fname, options['format'], options['th'])
        elif len(args) == 2:
            res_fname = args[0]
            pred_fname = args[1]
            eval_reranker(res_fname, pred_fname, options['format'], options['th'],
                          options['verbose'], options['reranking_th'], options['ignore_noanswer'])
        else:
            sys.exit(1)

    start_step = int(start_step)
    end_step = int(end_step)
    if(verbose == 'False'):
        verbose = False
    else:
        verbose = True
    if(ignore_noanswer == 'False'):
        ignore_noanswer = False
    else:
        ignore_noanswer = True
    options = {'th': th, 'reranking_th': reranking_th, 'format': op_format, 'verbose': verbose, 'ignore_noanswer': ignore_noanswer}
    for step in range(start_step, end_step + 1 , 200):
        args = [test_path + '/semeval/store/test-gold', test_path + '/semeval/store/test-' + run_id + '-' + str(step).zfill(6)]
        main(options, args)
