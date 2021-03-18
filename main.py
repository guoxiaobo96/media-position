from program.config import AnalysisArguments, get_config, DataArguments, MiscArgument, ModelArguments, AdapterArguments, TrainingArguments
from program.data import extract_data
from program.util import prepare_dirs_and_logger, save_config
from program.run_function import train_adapter, predict_adapter,analysis
from program.data_collect import twitter_collect


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
    elif misc_args.task == 'train_adapter':
        train_adapter(model_args, data_args, training_args, adapter_args)
    elif misc_args.task == 'predict_adapter':
        predict_adapter(misc_args, model_args, data_args, training_args, adapter_args)
    elif misc_args.task == 'analysis':
        analysis(misc_args, model_args, data_args, training_args, analysis_args)
    elif misc_args.task == "twitter_collect":
        twitter_collect(misc_args, data_args)



if __name__ == '__main__':
    misc_args, model_args, data_args, training_args, adapter_args, analysis_args = get_config()
    misc_args.global_debug = False
    prepare_dirs_and_logger(misc_args, model_args,
                            data_args, training_args, adapter_args, analysis_args)
    main(misc_args, model_args, data_args, training_args, adapter_args, analysis_args)
