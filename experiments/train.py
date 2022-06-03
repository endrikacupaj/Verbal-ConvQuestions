import time
import torch
import random
import logging
import numpy as np
from tqdm import tqdm
from transformers import AdamW
from torch.utils.data import DataLoader

from constants import *
from models import models
from utils import AverageMeter, save_checkpoint, evaluate
from data.conv_data import ConversationalAnswerVerbalizationData

logging.basicConfig(format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%d/%m/%Y %I:%M:%S %p',
                    level=logging.INFO,
                    handlers=[
                        logging.FileHandler(f'{args.path_results}/train_{args.model}.log', 'w'),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

random.seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

torch.cuda.set_device(args.cuda_device)

def main():
    dataset = ConversationalAnswerVerbalizationData()
    vocab = dataset.get_vocab()
    train_dataset, val_dataset, _ = dataset.get_data()

    model = models[args.model](vocab).to(DEVICE)

    logger.info(f'The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')

    optimizer = AdamW(model.parameters(), lr=args.lr)

    if args.resume:
        if os.path.isfile(args.resume):
            logger.info(f"=> loading checkpoint '{args.resume}''")
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint[EPOCH]
            best_val = checkpoint[BEST_VAL]
            model.load_state_dict(checkpoint[STATE_DICT])
            optimizer.load_state_dict(checkpoint[OPTIMIZER])
            logger.info(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint[EPOCH]})")
        else:
            logger.info(f"=> no checkpoint found at '{args.resume}'")
            best_val = float('inf')
    else:
        best_val = float('inf')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    logger.info(f'Model: {args.model}')
    logger.info(f'Domain: {args.domain}')
    logger.info(f'Training data len: {len(train_dataset)}')
    logger.info(f'Validation data len: {len(val_dataset)}')
    logger.info(f'Vocabulary length: {len(vocab)}')
    logger.info(f'Batch: {args.batch_size}')
    logger.info(f'Epochs: {args.epochs}')

    for epoch in range(args.start_epoch, args.epochs):
        train(train_loader, model, optimizer, epoch)

        if (epoch+1) % args.valfreq == 0:
            val_loss = evaluate(val_loader, model)
            if val_loss < best_val:
                best_val = min(val_loss, best_val)
                save_checkpoint({
                    EPOCH: epoch + 1,
                    STATE_DICT: model.state_dict(),
                    BEST_VAL: best_val,
                    OPTIMIZER: optimizer.state_dict(),
                    CURR_VAL: val_loss})
            logger.info(f'* Val loss: {val_loss:.4f}')

def train(train_loader, model, optimizer, epoch):
    losses = AverageMeter()
    model.train()

    pbar = tqdm(train_loader, desc=f'epoch: {epoch+1}, loss: {losses.avg:.4f}')
    for batch in pbar:
        optimizer.zero_grad()

        output = model(batch)

        losses.update(output[LOSS].data, batch[QUESTION_IDS].size(0))

        output[LOSS].backward()
        optimizer.step()

        pbar.set_description(f'epoch: {epoch+1}, loss: {losses.avg:.4f}')

if __name__ == '__main__':
    main()