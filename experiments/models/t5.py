import torch.nn as nn
from transformers import T5ForConditionalGeneration

from constants import *
from utils import init_weights

class T5(nn.Module):
    def __init__(self, vocab):
        super(T5, self).__init__()
        self.vocab = vocab

        self.t5 = T5ForConditionalGeneration.from_pretrained(T5_BASE)

    def forward(self, batch):
        output = self.t5(input_ids=batch[QUESTION_IDS], labels=batch[ANSWER_IDS])

        return {
            **batch,
            **{
                LOGITS: output.logits,
                LOSS: output.loss
            }
        }

    def predict(self, input_ids):
        self.eval()
        return self.t5.generate(input_ids).squeeze(0)
