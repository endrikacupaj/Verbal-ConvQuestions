import math
import torch
import random
import logging
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from constants import *
from models import models
from utils import evaluate, Predictor
from data.conv_data import ConversationalAnswerVerbalizationData

logging.basicConfig(format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%d/%m/%Y %I:%M:%S %p',
                    level=logging.INFO,
                    handlers=[
                        logging.FileHandler(f'{args.path_results}/test_{args.model}_{args.domain}.log', 'w'),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

random.seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

# set cuda device
torch.cuda.set_device(args.cuda_device)

def main():
    dataset = ConversationalAnswerVerbalizationData()
    vocab = dataset.get_vocab()
    _, _, test_dataset = dataset.get_data()

    model = models[args.model](vocab).to(DEVICE)

    logger.info(f'Model: {args.model}')
    logger.info(f'Domain: {args.domain}')
    logger.info(f"=> loading checkpoint '{args.model_path}'")
    if DEVICE.type==CPU:
        checkpoint = torch.load(f'{ROOT_PATH}/{args.model_path}', encoding='latin1', map_location=CPU)
    else:
        checkpoint = torch.load(f'{ROOT_PATH}/{args.model_path}', encoding='latin1')
    args.start_epoch = checkpoint[EPOCH]
    model.load_state_dict(checkpoint[STATE_DICT])
    logger.info(f"=> loaded checkpoint '{args.model_path}' (epoch {checkpoint[EPOCH]})")

    logger.info(f"Test data len: {len(test_dataset)}")

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    test_loss = evaluate(test_loader, model)
    test_perplexity = math.exp(test_loss)

    predictor = Predictor(model, dataset.tokenizer)
    predictor.predict(test_dataset)

    logger.info(f'Test results:')
    logger.info(f'** Loss: {test_loss:.4f}')
    logger.info(f'** PPL: {test_perplexity:.4f}')

    for k in predictor.bleu_score_meter.keys():
        logger.info(f'** {k.upper()}')
        logger.info(f'**** BLEU-1 score: {predictor.bleu_score_meter[k][BLEU_1].avg:.4f}')
        logger.info(f'**** BLEU-2 score: {predictor.bleu_score_meter[k][BLEU_2].avg:.4f}')
        logger.info(f'**** BLEU-3 score: {predictor.bleu_score_meter[k][BLEU_3].avg:.4f}')
        logger.info(f'**** BLEU-4 score: {predictor.bleu_score_meter[k][BLEU_4].avg:.4f}')
        logger.info(f'**** METEOR score: {predictor.meteor_score_meter[k].avg:.4f}')

if __name__ == '__main__':
    main()