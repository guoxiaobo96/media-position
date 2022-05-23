from program.config import AnalysisArguments, get_config, DataArguments, MiscArgument, ModelArguments, TrainingArguments, DataAugArguments, PredictArguments
from program.data import extract_data
from program.util import prepare_dirs_and_logger
from program.run_function import train_lm, predict_token, label_score_analysis, data_augemnt, eval_lm, train_classifier, label_masked_token, encode_media
from program.data_collect import data_collect


def main(
        misc_args: MiscArgument,
        model_args: ModelArguments,
        data_args: DataArguments,
        aug_args: DataAugArguments,
        training_args: TrainingArguments,
        analysis_args: AnalysisArguments,
        predict_args: PredictArguments
) -> None:
    if misc_args.task == 'extract_data':
        extract_data(misc_args, data_args)
    elif misc_args.task == 'train_lm':
        train_lm(model_args, data_args, training_args)
    elif misc_args.task == 'eval_lm':
        eval_lm(model_args, data_args, training_args)
    elif misc_args.task == 'train_classifier':
        train_classifier(model_args, data_args, training_args)
    elif misc_args.task == 'label_masked_token':
        label_masked_token(misc_args, model_args, data_args, training_args)
    elif misc_args.task == 'label_score_predict':
        predict_token(misc_args, model_args, data_args,
                            training_args, predict_args)
    elif misc_args.task == 'label_score_analysis':
        # label_score_analysis(misc_args, model_args,
        #                      data_args, training_args, analysis_args, predict_args, 'SoA-t')
        # label_score_analysis(misc_args, model_args,
        #                      data_args, training_args, analysis_args, predict_args, 'SoA-s')
        # label_score_analysis(misc_args, model_args,
        #                      data_args, training_args, analysis_args, predict_args, 'MBR')
        label_score_analysis(misc_args, model_args,
                             data_args, training_args, analysis_args, predict_args, 'human')
    elif misc_args.task == "data_collect":
        if aug_args.augment_type == 'original':
            data_collect(misc_args, data_args)
        else:
            data_augemnt(misc_args, data_args, aug_args)
    elif misc_args.task == "encode_media":
        encode_media(misc_args, model_args, data_args, training_args)


if __name__ == '__main__':
    misc_args, model_args, data_args, aug_args, training_args, analysis_args, baseline_args = get_config()
    # misc_args.global_debug = False
    prepare_dirs_and_logger(misc_args, model_args,
                            data_args, training_args, analysis_args, baseline_args)
    main(misc_args, model_args, data_args, aug_args,
         training_args, analysis_args, baseline_args)
