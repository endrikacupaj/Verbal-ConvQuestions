import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='Answer Verbalization for Conversational Question Answering over Knowledge Graphs')

    # general
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--cuda_device', default=0, type=int)

    # models
    parser.add_argument('--model', default='transformer', choices=['convolutional',
                                                                   'transformer',
                                                                   'bert', 'bart', 't5'], type=str)

    # model parameters
    parser.add_argument('--emb_dim', default=512, type=int)
    parser.add_argument('--dropout', default=0.1, type=int)
    parser.add_argument('--heads', default=8, type=int)
    parser.add_argument('--layers', default=2, type=int)
    parser.add_argument('--max_positions', default=300, type=int)
    parser.add_argument('--max_input_size', default=30, type=int)
    parser.add_argument('--pf_dim', default=300, type=int)

    # training
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--valfreq', default=1, type=int)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--clip', default=1, type=int)
    parser.add_argument('--batch_size', default=16, type=int)

    # test and experiments
    parser.add_argument('--snapshots', default='experiments/snapshots', type=str)
    parser.add_argument('--path_results', default='experiments/results', type=str)
    parser.add_argument('--model_path', default='experiments/snapshots/', type=str)

    # domain
    parser.add_argument('--domain', default='all', choices=['all',
                                                            'books'
                                                            'music',
                                                            'movies',
                                                            'tv_series',
                                                            'soccer'], type=str)

    return parser