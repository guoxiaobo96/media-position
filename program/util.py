from genericpath import exists
import os
import logging

from .config import DataArguments, MiscArgument, ModelArguments, TrainingArguments, AnalysisArguments, BaselineArguments


def prepare_dirs_and_logger(
        misc_args: MiscArgument,
        model_args: ModelArguments,
        data_args: DataArguments,
        training_args: TrainingArguments,
        analysis_args: AnalysisArguments,
        baseline_args: BaselineArguments

) -> None:
    # os.chdir(os.path.dirname(__file__))
    os.chdir(misc_args.root_dir)
    path = os.getcwd()

    if not os.path.exists(misc_args.log_dir):
        os.makedirs(misc_args.log_dir)

    # if not os.path.exists(analysis_args.analysis_result_dir):
    #     os.makedirs(analysis_args.analysis_result_dir)

    formatter = logging.Formatter("%(asctime)s:%(levelname)s::%(message)s")
    logger = logging.getLogger()

    for hdlr in logger.handlers:
        logger.removeHandler(hdlr)

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger.addHandler(handler)
