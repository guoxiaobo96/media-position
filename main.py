from program.config import AnalysisArguments, get_config, DataArguments, MiscArgument, ModelArguments,TrainingArguments, DataAugArguments, BaselineArguments
from program.data import extract_data
from program.util import prepare_dirs_and_logger, save_config
from program.run_function import train_lm, analysis, label_score_predict, label_score_analysis, data_augemnt, train_mask_score_model, eval_lm,  generate_baseline, train_classifier, label_masked_token
from program.data_collect import twitter_collect, article_collect, data_collect


def main(
        misc_args: MiscArgument,
        model_args: ModelArguments,
        data_args: DataArguments,
        aug_args: DataAugArguments,
        training_args: TrainingArguments,
        analysis_args: AnalysisArguments,
        baseline_args: BaselineArguments
) -> None:
    if misc_args.task == 'extract_data':
        extract_data(misc_args, data_args)
    elif misc_args.task == 'generate_baseline':
        generate_baseline(misc_args,baseline_args,data_args)
    elif misc_args.task == 'train_lm':
        train_lm(model_args, data_args, training_args)
    elif misc_args.task == 'eval_lm':
        eval_lm(model_args, data_args, training_args)
    elif misc_args.task == 'train_classifier':
        train_classifier(model_args, data_args, training_args)
    elif misc_args.task == 'label_masked_token':
        label_masked_token(misc_args, model_args, data_args, training_args)
    elif misc_args.task == 'train_mask_score_model':
        train_mask_score_model(model_args, data_args,
                               training_args, analysis_args)
    elif misc_args.task == 'label_score_predict':
        label_score_predict(misc_args, model_args, data_args,
                            training_args)
    elif misc_args.task == 'label_score_analysis':
        label_score_analysis(misc_args, model_args,
                             data_args, training_args, analysis_args)
    elif misc_args.task == 'analysis':
        analysis(misc_args, model_args, data_args,
                 training_args, analysis_args)
    elif misc_args.task == "data_collect":
        if aug_args.augment_type == 'original':
            data_collect(misc_args, data_args)
        else:
            data_augemnt(misc_args, data_args, aug_args)

if __name__ == '__main__':
    misc_args, model_args, data_args, aug_args, training_args, analysis_args, baseline_args = get_config()
    # misc_args.global_debug = False
    prepare_dirs_and_logger(misc_args, model_args,
                            data_args, training_args, analysis_args, baseline_args)
    main(misc_args, model_args, data_args, aug_args, 
         training_args,analysis_args,baseline_args)
