import torch.nn as nn
from transformers import BertGenerationEncoder, BertGenerationDecoder, EncoderDecoderModel

from constants import *
from utils import init_weights

class Bert(nn.Module):
    def __init__(self, vocab):
        super(Bert, self).__init__()
        self.vocab = vocab
        self.encoder = BertGenerationEncoder.from_pretrained(BERT_BASE)
        self.decoder = BertGenerationDecoder.from_pretrained(BERT_BASE, add_cross_attention=True, is_decoder=True)

        self.bert = EncoderDecoderModel(encoder=self.encoder, decoder=self.decoder)

    def forward(self, batch):
        output = self.bert(input_ids=batch[QUESTION_IDS], decoder_input_ids=batch[ANSWER_IDS], labels=batch[ANSWER_IDS])

        return {
            **batch,
            **{
                LOGITS: output.logits,
                LOSS: output.loss
            }
        }

    def predict(self, input_ids):
        self.eval()
        return self.bert.generate(input_ids).squeeze(0)
