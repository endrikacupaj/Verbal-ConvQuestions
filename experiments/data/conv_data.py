import os
import re
import glob
import json
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from constants import *
from models import tokenizers

class PtDataset(Dataset):
    def __init__(self, questions, answers, domains):
        self.question_ids = torch.LongTensor(questions.data['input_ids']).to(DEVICE)
        self.question_attention = torch.LongTensor(questions.data['attention_mask']).to(DEVICE)
        self.answer_ids = torch.LongTensor(answers.data['input_ids']).to(DEVICE)
        self.answer_attention = torch.LongTensor(answers.data['attention_mask']).to(DEVICE)
        self.domains = domains

    def __getitem__(self, idx):
        return {
            QUESTION_IDS: self.question_ids[idx],
            QUESTION_ATTENTION: self.question_attention[idx],
            ANSWER_IDS: self.answer_ids[idx],
            ANSWER_ATTENTION: self.answer_attention[idx],
            DOMAIN: self.domains[idx],
        }

    def __len__(self):
        return len(self.question_ids)

class ConversationalAnswerVerbalizationData:
    ANSWER_REGEX = r'\[.*?\]'

    def __init__(self):
        self.data_path = str(ROOT_PATH) + '/data/Verbal-ConvQuestions'
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizers[args.model])
        self.load_data()

    def _cover_answer(self, text, ans_ent=ANSWER_REGEX, ans_token=ANS_TOKEN):
        try:
            return re.sub(ans_ent, ans_token, text)
        except:
            return text

    def prepare_data(self, data):
        data_question, data_answer, data_domain = [], [], []
        for d in data:
            prev_utterances = []
            domain = d['domain']
            for q in d[QUESTIONS]:
                question = q[QUESTION]
                answer = self._cover_answer(q[VERBALIZED_ANSWER])

                # data_question.append(question)
                data_question.append(f' {SEP_TOKEN} '.join(prev_utterances + [question]))
                data_answer.append(answer)
                data_domain.append(domain)

                prev_utterances.extend([question, answer])

        return {
            QUESTION: self.tokenizer(data_question, truncation=True, padding=True),
            ANSWER: self.tokenizer(data_answer, truncation=True, padding=True),
            DOMAIN: data_domain
        }

    def load_data(self):
        train, test, val = [], [], []
        if args.domain is ALL:
            for train_file in glob.glob(f'{self.data_path}/train/*.json'):
                with open(train_file) as json_file:
                    train.extend(json.load(json_file))

            for val_file in glob.glob(f'{self.data_path}/val/*.json'):
                with open(val_file) as json_file:
                    val.extend(json.load(json_file))

            for test_file in glob.glob(f'{self.data_path}/test/*.json'):
                with open(test_file) as json_file:
                    test.extend(json.load(json_file))
        else:
            with open(f'{self.data_path}/train/train_{args.domain}.json') as json_file:
                train = json.load(json_file)

            with open(f'{self.data_path}/val/val_{args.domain}.json') as json_file:
                val = json.load(json_file)

            with open(f'{self.data_path}/test/test_{args.domain}.json') as json_file:
                test = json.load(json_file)

        self.train_data = self.prepare_data(train)
        self.val_data = self.prepare_data(val)
        self.test_data = self.prepare_data(test)

        self.train_dataset = PtDataset(self.train_data[QUESTION], self.train_data[ANSWER], self.train_data[DOMAIN])
        self.val_dataset = PtDataset(self.val_data[QUESTION], self.val_data[ANSWER], self.val_data[DOMAIN])
        self.test_dataset = PtDataset(self.test_data[QUESTION], self.test_data[ANSWER], self.test_data[DOMAIN])

    def get_data(self):
        return self.train_dataset, self.val_dataset, self.test_dataset

    def get_vocab(self):
        return self.tokenizer.get_vocab()