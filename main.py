from program.config import AnalysisArguments, get_config, DataArguments, MiscArgument, ModelArguments, AdapterArguments, TrainingArguments
from program.data import extract_data
from program.util import prepare_dirs_and_logger, save_config
from program.run_function import mlm_train, analysis, label_score_predict, label_score_analysis, data_augemnt, train_mask_score_model, mlm_eval, sentence_replacement_train
from program.data_collect import twitter_collect, article_collect, data_collect


def main(
        misc_args: MiscArgument,
        model_args: ModelArguments,
        data_args: DataArguments,
        training_args: TrainingArguments,
        adapter_args: AdapterArguments,
        analysis_args: AnalysisArguments
) -> None:
    if misc_args.task == 'extract_data':
        extract_data(misc_args, data_args)
    elif misc_args.task == 'train_mlm':
        mlm_train(model_args, data_args, training_args, adapter_args)
    elif misc_args.task == 'eval_mlm':
        mlm_eval(model_args, data_args, training_args, adapter_args)
    elif misc_args.task == 'train_sentence_replacement':
        sentence_replacement_train(
            model_args, data_args, training_args, adapter_args)
    elif misc_args.task == 'train_mask_score_model':
        train_mask_score_model(model_args, data_args,
                               training_args, analysis_args, adapter_args)
    elif misc_args.task == 'label_score_predict':
        label_score_predict(misc_args, model_args, data_args,
                            training_args, adapter_args)
    elif misc_args.task == 'label_score_analysis':
        label_score_analysis(misc_args, model_args,
                             data_args, training_args, analysis_args)
    elif misc_args.task == 'analysis':
        analysis(misc_args, model_args, data_args,
                 training_args, analysis_args)
    elif misc_args.task == "data_collect":
        if data_args.data_type == 'original':
            data_collect(misc_args, data_args)
        else:
            data_augemnt(misc_args, data_args)

if __name__ == '__main__':
    misc_args, model_args, data_args, training_args, adapter_args, analysis_args = get_config()
    # misc_args.global_debug = False
    prepare_dirs_and_logger(misc_args, model_args,
                            data_args, training_args, adapter_args, analysis_args)
    main(misc_args, model_args, data_args,
         training_args, adapter_args, analysis_args)
