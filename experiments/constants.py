import os
import torch
from pathlib import Path
from args import get_parser
from accelerate import Accelerator

# set root path
ROOT_PATH = Path(os.path.dirname(__file__))

# read parser
parser = get_parser()
args = parser.parse_args()

# define device
CUDA = 'cuda'
CPU = 'cpu'
DEVICE = torch.device(CUDA if torch.cuda.is_available() else CPU)

TRAIN = 'train'
VAL = 'val'
TEST = 'test'

# helper tokens
ENT_TOKEN = '<ent>'
ANS_TOKEN = '<ans>'
SEP_TOKEN = '<sep>'

# models
BERT_BASE = 'bert-base-uncased'
BART_BASE = 'facebook/bart-base'
T5_BASE = 't5-base'

CONVOLUTIONAL = 'convolutional'
TRANSFORMER = 'transformer'
BERT = 'bert'
ROBERTA = 'roberta'
BART = 'bart'
T5_MODEL = 't5'


# training
EPOCH = 'epoch'
STATE_DICT = 'state_dict'
BEST_VAL = 'best_val'
OPTIMIZER = 'optimizer'
CURR_VAL = 'curr_val'
LOSS = 'loss'
LOGITS = 'logits'

# model
TARGET = 'target'
QUESTION_IDS = 'question_ids'
QUESTION_ATTENTION = 'question_attention'
ANSWER_IDS = 'answer_ids'
ANSWER_ATTENTION = 'answer_attention'
PARAPHRASED_QUESTION = 'paraphrased_question'
PARAPHRASED_ANSWER = 'paraphrased_answer'

# domain
DOMAIN = 'domain'
ALL = 'all'
BOOKS = 'books'
MUSIC = 'music'
MOVIES = 'movies'
TV_SERIES = 'tv_series'
SOCCER = 'soccer'

# other
QUESTION = 'question'
ANSWER = 'answer'
QUESTIONS = 'questions'
VERBALIZED_ANSWER = 'verbalized_answer'
RESULTS = 'results'
REFERENCE = 'reference'
HYPOTHESIS = 'hypothesis'
BLEU_SCORE = 'bleu_score'
BLEU_1 = 'bleu_1'
BLEU_2 = 'bleu_2'
BLEU_3 = 'bleu_3'
BLEU_4 = 'bleu_4'
METEOR_SCORE = 'meteor_score'