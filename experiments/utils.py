from __future__ import division
import os
import re
import json
import nltk
import glob
import torch
import logging
import torch.nn as nn
from tqdm import tqdm

from constants import *

logger = logging.getLogger(__name__)

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Predictor(object):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

        self.results = []
        self.bleu_score_meter = {
            ALL: {
                BLEU_1: AverageMeter(),
                BLEU_2: AverageMeter(),
                BLEU_3: AverageMeter(),
                BLEU_4: AverageMeter()
            }
        }
        self.meteor_score_meter = {
            ALL: AverageMeter()
        }

    def _bleu_score(self, reference, hypothesis):
        return {
            BLEU_1: nltk.translate.bleu_score.sentence_bleu([reference], hypothesis, weights=(1.0, 0.0, 0.0, 0.0)),
            BLEU_2: nltk.translate.bleu_score.sentence_bleu([reference], hypothesis, weights=(0.5, 0.5, 0.0, 0.0)),
            BLEU_3: nltk.translate.bleu_score.sentence_bleu([reference], hypothesis, weights=(0.33, 0.33, 0.33, 0.0)),
            BLEU_4: nltk.translate.bleu_score.sentence_bleu([reference], hypothesis, weights=(0.25, 0.25, 0.25, 0.25)),
        }

    def _meteor_score(self, reference, hypothesis):
        return nltk.translate.meteor_score.single_meteor_score(' '.join(reference), ' '.join(hypothesis))

    def predict(self, data):
        for d in tqdm(data):
            domain = d[DOMAIN]
            question_ids, answer_ids = d[QUESTION_IDS], d[ANSWER_IDS]
            prediction_ids = self.model.predict(question_ids.unsqueeze(0))

            reference = self.tokenizer.convert_tokens_to_string(
                self.tokenizer.convert_ids_to_tokens(answer_ids, skip_special_tokens=True)
            ).lower().replace("?", " ?").replace(".", " .").replace(",", " ,").replace("'", " '").split()

            hypothesis = self.tokenizer.convert_tokens_to_string(
                self.tokenizer.convert_ids_to_tokens(prediction_ids, skip_special_tokens=True)
            ).lower().replace("?", " ?").replace(".", " .").replace(",", " ,").replace("'", " '").split()

            blue_score = self._bleu_score(reference, hypothesis)
            meteor_score = self._meteor_score(reference, hypothesis)

            if domain not in self.bleu_score_meter:
                self.bleu_score_meter[domain] = {
                    BLEU_1: AverageMeter(),
                    BLEU_2: AverageMeter(),
                    BLEU_3: AverageMeter(),
                    BLEU_4: AverageMeter()
                }
                self.meteor_score_meter[domain] = AverageMeter()

            # update domain scores
            self.bleu_score_meter[domain][BLEU_1].update(blue_score[BLEU_1])
            self.bleu_score_meter[domain][BLEU_2].update(blue_score[BLEU_2])
            self.bleu_score_meter[domain][BLEU_3].update(blue_score[BLEU_3])
            self.bleu_score_meter[domain][BLEU_4].update(blue_score[BLEU_4])
            self.meteor_score_meter[domain].update(meteor_score)

            if args.domain == ALL:
                # update all scores
                self.bleu_score_meter[ALL][BLEU_1].update(blue_score[BLEU_1])
                self.bleu_score_meter[ALL][BLEU_2].update(blue_score[BLEU_2])
                self.bleu_score_meter[ALL][BLEU_3].update(blue_score[BLEU_3])
                self.bleu_score_meter[ALL][BLEU_4].update(blue_score[BLEU_4])
                self.meteor_score_meter[ALL].update(meteor_score)

            self.results.append({
                REFERENCE: reference,
                HYPOTHESIS: hypothesis,
                BLEU_SCORE: {
                    BLEU_1: blue_score[BLEU_1],
                    BLEU_2: blue_score[BLEU_2],
                    BLEU_3: blue_score[BLEU_3],
                    BLEU_4: blue_score[BLEU_4]
                },
                METEOR_SCORE: meteor_score
            })

    def write_results(self):
        save_dict = json.dumps(self.results, indent=4)
        save_dict_no_space_1 = re.sub(r'": \[\s+', '": [', save_dict)
        save_dict_no_space_2 = re.sub(r'",\s+', '", ', save_dict_no_space_1)
        save_dict_no_space_3 = re.sub(r'"\s+\]', '"]', save_dict_no_space_2)
        with open(f'{ROOT_PATH}/{args.path_results}/{args.model}_results.json', 'w', encoding='utf-8') as json_file:
            json_file.write(save_dict_no_space_3)

    def reset(self):
        self.results = []
        self.bleu_score_meter = {
            ALL: {
                BLEU_1: AverageMeter(),
                BLEU_2: AverageMeter(),
                BLEU_3: AverageMeter(),
                BLEU_4: AverageMeter()
            }
        }
        self.meteor_score_meter = {
            ALL: AverageMeter()
        }

def evaluate(loader, model):
    losses = AverageMeter()
    model.eval()
    pbar = tqdm(loader, desc=f'loss: {losses.avg:.4f}')
    with torch.no_grad():
        for batch in pbar:
            output = model(batch)
            losses.update(output[LOSS].data, batch[QUESTION_IDS].size(0))
            pbar.set_description(f'loss: {losses.avg:.4f}')

    return losses.avg

def init_weights(model):
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

def save_checkpoint(state):
    for f in glob.glob(f'{ROOT_PATH}/{args.snapshots}/{args.model}_{args.domain}*'):
        os.remove(f)
    filename = f'{ROOT_PATH}/{args.snapshots}/{args.model}_{args.domain}_e{state[EPOCH]}_v{state[CURR_VAL]:.4f}.pth.tar'
    torch.save(state, filename)