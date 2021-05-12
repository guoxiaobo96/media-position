import os


from transformers.trainer_utils import default_compute_objective
import transformers
from transformers import (
    MODEL_WITH_LM_HEAD_MAPPING,
    AdapterArguments,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
    logging
)
import argparse
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any, NewType


MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
logging.set_verbosity_error()

@dataclass
class MiscArgument:
    """
    Arguments pertrain to the misc arguments about the run environment of the program
    """
    task: str = field(
        metadata={"help": "The task of running"}
    )
    # target: str = field(
    #     metadata={"help": "The task of running"}
    # )
    root_dir: str = field(
        default='/home/xiaobo/media-position', metadata={"help": "The relative path to the root dir"}
    )
    log_dir: Optional[str] = field(
        default='log', metadata={"help": "The relative path to the log dir"}
    )
    gpu_id: str = field(
        default='0', metadata={"help": "The id of gpu which runs the work"}
    )

    load_model: bool = field(
        default=False, metadata={"help": "Whether to load the trained model"}
    )

    global_debug: bool = field(
        default=False, metadata={"help": "Whether the program is in debug mode"}
    )

@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    original_data_dir: str = field(
        default="/data/xiaobo/media-position/data_original", metadata={"help": "The dir of original data"}
    )
    
    data_dir: str = field(
        default="/data/xiaobo/media-position/data", metadata={"help": "The dir of processed data"}
    )

    dataset: str = field(
        default='', metadata={"help": "The dataset of train and eval data"}
    )

    data_topic: str = field(
        default='', metadata={"help": "The topic of train and eval data"}
    )

    data_type: str = field(
        default='', metadata={"help": "The data type of train and eval data should be twitter or article"}
    )

    data_path: str = field(
        default='', metadata={"help": "The relative path of the data the dataset"}
    )
    train_data_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    train_data_files: Optional[str] = field(
        default=None,
        metadata={
            "help": "The input training data files (multiple files in glob format). "
            "Very often splitting large files to smaller files can prevent tokenizer going out of memory"
        },
    )
    eval_data_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    train_ref_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input train ref data file for whole word mask in Chinese."},
    )
    eval_ref_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input eval ref data file for whole word mask in Chinese."},
    )
    line_by_line: bool = field(
        default=False,
        metadata={
            "help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )

    mlm: bool = field(
        default=False, metadata={"help": "Train with masked-language modeling loss instead of language modeling."}
    )
    whole_word_mask: bool = field(default=False, metadata={
                                  "help": "Whether ot not to use whole word mask."})
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    plm_probability: float = field(
        default=1 / 6,
        metadata={
            "help": "Ratio of length of a span of masked tokens to surrounding context length for permutation language modeling."
        },
    )
    max_span_length: int = field(
        default=5, metadata={"help": "Maximum length of a span of masked tokens for permutation language modeling."}
    )

    block_size: int = field(
        default=-1,
        metadata={
            "help": "Optional input sequence length after tokenization."
            "The training dataset will be truncated in block of this size for training."
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

@dataclass
class AnalysisArguments:
    analysis_data_dir: str = field(
        default="log", metadata={"help": "The dir of analysis data"}
    )
    analysis_result_dir: str = field(
        default="analysis", metadata={"help": "The dir of analysis result"}
    )
    analysis_data_type: str = field(
        default="full", metadata={"help": "The type of data for analyzing (dataset, country and full)"}
    )
    analysis_encode_method: str = field(
        default="term", metadata={"help": "The method for encoding predicted result"}
    )
    analysis_compare_method: str = field(
        default="cluster", metadata={"help": "The method for comparing the result, can choose from cluster and distance"}
    )
    analysis_cluster_method: Optional[str] = field(
        default="DBSCAN", metadata={"help": "The method for clustering"}
    )
    analysis_distance_method: Optional[str] = field(
        default="Cosine", metadata={"help": "The method for calculating distance"}
    )
    graph_distance: Optional[str] = field(
        default="", metadata={"help": "The distance used for creating graph"}
    )
    graph_kernel: Optional[str] = field(
        default="", metadata={"help": "The kernel used for calculating difference"}
    )    

# @dataclass
# class ArticleMap:
#     dataset_to_name: Dict = field(default_factory=lambda: {'ABC.com':'ABC News','BBC':'BBC','Breitbart':'Breitbart','CBS':'CBS News','CNN':'CNN','Fox':'Fox News','guardiannews.com':'Guardian','HuffPost':'HuffPost','NPR':'NPR','NYtimes':'New York Times','rushlimbaugh.com':'Rush Limbaugh Show (radio)','sean':'Sean Hannity Show (radio)','usatoday':'USA Today','wallstreet':'Wall Street Journal','washington':'Washington Post'})
#     name_to_dataset: Dict = field(init=False)
#     dataset_list: List[str] = field(init=False)
#     left_dataset_list: List[str] = field(default_factory=lambda:['Breitbart', 'Fox', 'sean','rushlimbaugh.com'])

#     def __post_init__(self):
#         self.name_to_dataset = {v: k for k, v in self.dataset_to_name.items()}
#         self.dataset_list = [k for k,v in self.dataset_to_name.items()]

@dataclass
class ArticleMap:
    dataset_to_name: Dict = field(default_factory=lambda: {'Breitbart':'Breitbart','CBS':'CBS News','CNN':'CNN'})
    name_to_dataset: Dict = field(init=False)
    dataset_list: List[str] = field(init=False)
    left_dataset_list: List[str] = field(default_factory=lambda:['Breitbart', 'Fox', 'sean','rushlimbaugh.com'])

    def __post_init__(self):
        self.name_to_dataset = {v: k for k, v in self.dataset_to_name.items()}
        self.dataset_list = [k for k,v in self.dataset_to_name.items()]


@dataclass
class TwitterMap:
    dataset_to_name: Dict = field(default_factory=lambda: {'BreitbartNews':'Breitbart','CNN':'CNN','FoxNews':'Fox News','nytimes':'New York Times','seanhannity':'Sean Hannity Show (radio)','washingtonpost':'Washington Post'})
    name_to_dataset: Dict = field(init=False)
    dataset_list: List[str] = field(init=False)
    left_dataset_list: List[str] = field(default_factory=lambda:['BreitbartNews', 'FoxNews', 'seanhannity'])

    def __post_init__(self):
        self.name_to_dataset = {v: k for k, v in self.dataset_to_name.items()}
        self.dataset_list = [k for k,v in self.dataset_to_name.items()]

@dataclass
class SourceMap:
    republican_datasets_list: List[str] = field(
        default_factory=lambda: ['Fox News', 'Sean Hannity Show (radio)', 'Breitbart'])
    democrat_datasets_list: List[str] = field(
        default_factory=lambda: ['CNN', 'New York Times', 'NPR'])

    dataset_to_name: Dict = field(default_factory=lambda: {
                                  'FoxNews': 'Fox News', 'seanhannity': 'Sean Hannity Show (radio)', 'BreitbartNews': 'Breitbart', 'CNN': 'CNN', 'nytimes': 'New York Times', 'NPR': 'NPR'})

    name_to_dataset: Dict = field(init=False)
    position_to_name: Dict = field(init=False)
    full_datasets_list: List[str] = field(init=False)
    position_list: List[str] = field(
        default_factory=lambda: ['republican', 'democrat'])

    def __post_init__(self):
        self.full_datasets_list = self.democrat_datasets_list + self.republican_datasets_list

        self.position_to_name = {
            'Republican': self.republican_datasets_list, 'Democrat': self.democrat_datasets_list}

        self.name_to_dataset = {v: k for k, v in self.dataset_to_name.items()}

@dataclass
class TrustMap:
    republican_datasets_list: List[str] = field(
        default_factory=lambda: ['Fox News', 'Sean Hannity Show (radio)', 'Breitbart'])
    democrat_datasets_list: List[str] = field(
        default_factory=lambda: ['CNN', 'New York Times', 'Washington Post', 'MSNBC', 'NBC NEWS', 'NPR'])

    dataset_to_name: Dict = field(default_factory=lambda: {
                                  'FoxNews': 'Fox News', 'seanhannity': 'Sean Hannity Show (radio)', 'BreitbartNews': 'Breitbart', 'CNN': 'CNN', 'nytimes': 'New York Times', 'washingtonpost': 'Washington Post','MSNBC':'MSNBC','NBCNews':'NBC NEWS','NPR':'NPR'})

    name_to_dataset: Dict = field(init=False)
    position_to_name: Dict = field(init=False)
    full_datasets_list: List[str] = field(init=False)
    position_list: List[str] = field(
        default_factory=lambda: ['republican', 'democrat'])

    def __post_init__(self):
        self.full_datasets_list = self.democrat_datasets_list + self.republican_datasets_list

        self.position_to_name = {
            'Republican': self.republican_datasets_list, 'Democrat': self.democrat_datasets_list}

        self.name_to_dataset = {v: k for k, v in self.dataset_to_name.items()}



def get_config() -> Tuple:

    def _get_config(
        misc_args: MiscArgument,
        data_args: DataArguments,
        model_args: ModelArguments,
        training_args: TrainingArguments,
        adapter_args: AdapterArguments,
        analysis_args: AnalysisArguments
    ) -> None:
    
        data_args.original_data_dir = os.path.join(
            data_args.original_data_dir, data_args.data_type)
        data_args.data_path = os.path.join(
            data_args.data_dir, os.path.join(data_args.dataset, data_args.data_type))
        training_args.output_dir = os.path.join(
            training_args.output_dir, os.path.join(data_args.dataset, data_args.data_type))
        if training_args.do_train:
            data_args.train_data_file = os.path.join(
                data_args.data_path, adapter_args.language+'.train')
        if training_args.do_eval:
            data_args.eval_data_file = os.path.join(
                data_args.data_path, adapter_args.language+'.valid')
        if misc_args.load_model:
            if adapter_args.load_adapter == '':
                adapter_args.load_adapter = os.path.join(
                    training_args.output_dir, adapter_args.language)
            adapter_args.adapter_config = os.path.join(
                adapter_args.load_adapter, 'adapter_config.json')

        analysis_args.analysis_data_dir = os.path.join(os.path.join(
            analysis_args.analysis_data_dir, data_args.data_type), 'json')
        analysis_args.analysis_result_dir = os.path.join(os.path.join(os.path.join(
            analysis_args.analysis_result_dir, data_args.data_type), analysis_args.analysis_compare_method), analysis_args.analysis_data_type)
        
        training_args.disable_tqdm=False

    parser = HfArgumentParser((MiscArgument, DataArguments,
                               ModelArguments, TrainingArguments, AdapterArguments, AnalysisArguments))

    misc_args, data_args, model_args, training_args, adapter_args, analysis_args = parser.parse_args_into_dataclasses()
    _get_config(misc_args, data_args, model_args,
                training_args, adapter_args, analysis_args)
    set_seed(training_args.seed)
    return misc_args, model_args, data_args, training_args, adapter_args, analysis_args

if __name__=='__main__':
    get_config()