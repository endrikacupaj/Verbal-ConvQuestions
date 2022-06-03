from constants import *

from models.transformer import Transformer
from models.convolutional import Convolutional
from models.bert import Bert
from models.bart import Bart
from models.t5 import T5

models = {
    CONVOLUTIONAL: Convolutional,
    TRANSFORMER: Transformer,
    BERT: Bert,
    BART: Bart,
    T5_MODEL: T5
}

tokenizers = {
    CONVOLUTIONAL: BERT_BASE,
    TRANSFORMER: BERT_BASE,
    BERT: BERT_BASE,
    BART: BART_BASE,
    T5_MODEL: T5_BASE
}