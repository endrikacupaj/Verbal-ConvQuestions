import torch.nn as nn
from transformers import BartForConditionalGeneration

from constants import *
from utils import init_weights

class Bart(nn.Module):
    def __init__(self, vocab):
        super(Bart, self).__init__()
        self.vocab = vocab

        self.bart = BartForConditionalGeneration.from_pretrained(BART_BASE)

    def forward(self, batch):
        output = self.bart(input_ids=batch[QUESTION_IDS], labels=batch[ANSWER_IDS])

        return {
            **batch,
            **{
                LOGITS: output.logits,
                LOSS: output.loss
            }
        }

    def predict(self, input_ids):
        self.eval()
        return self.bart.generate(input_ids).squeeze(0)
