import glob
import json
from constants import *

TOKENIZE_SEQ = lambda x: x.replace("?", " ?").\
                                     replace(".", " .").\
                                     replace(",", " ,").\
                                     replace("'", " '").\
                                     split()

train, test, val = [], [], []

data_path = str(ROOT_PATH) + '/data/Verbal-ConvQuestions'

for train_file in glob.glob(f'{data_path}/train/*.json'):
    with open(train_file) as json_file:
        train.extend(json.load(json_file))

for val_file in glob.glob(f'{data_path}/val/*.json'):
    with open(val_file) as json_file:
        val.extend(json.load(json_file))

for test_file in glob.glob(f'{data_path}/test/*.json'):
    with open(test_file) as json_file:
        test.extend(json.load(json_file))

for data in [(train, 'train'), (val, 'val'), (test, 'test')]:
    count_para_questions, count_para_answers = 0, 0
    question_length, answer_length = [], []
    for d in data[0]:
        for q in d[QUESTIONS]:
            question_length.append(len(TOKENIZE_SEQ(q[QUESTION])))
            answer_length.append(len(TOKENIZE_SEQ(q[VERBALIZED_ANSWER])))

            for p_q in q[PARAPHRASED_QUESTION]:
                count_para_questions += 1
                question_length.append(len(TOKENIZE_SEQ(p_q)))

            for p_a in q[PARAPHRASED_QUESTION]:
                count_para_answers += 1
                answer_length.append(len(TOKENIZE_SEQ(p_a)))

    print(f'Stats for set {data[1]}')
    print(f'# Conversations {len(data[0])}')
    print(f'# Paraprased Questions {count_para_questions}')
    print(f'# Paraprased Answers {count_para_answers}')
    print(f'Avg. Question length {sum(question_length)/len(question_length)}')
    print(f'Avg. Answer length {sum(answer_length)/len(question_length)}')