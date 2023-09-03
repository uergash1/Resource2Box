import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='fedweb14', type=str, help='dataset')
    parser.add_argument("--title", default=True, type=bool, help='include document title')
    parser.add_argument("--body", default=True, type=bool, help='include document body')
    parser.add_argument("--train_pair_count", default=10, type=int, help='number of train pairs')
    parser.add_argument("--train", default=40, type=int, help='number of train queries')
    parser.add_argument("--test", default=10, type=int, help='number of test queries')
    parser.add_argument("--random_seed", default=10, type=int, help='random seed')
    parser.add_argument("--gpu", default='0', type=str, help='gpu number')

    parser.add_argument("--batch_size", default=64, type=int, help='batch size')
    parser.add_argument("--epochs", default=3, type=int, help='number of epochs')
    parser.add_argument("--learning_rate", default=1e-3, type=float, help='learning rate')
    parser.add_argument("--dim", default=768, type=int, help='embedding dimension')

    parser.add_argument("--box_type", default="geometric", type=str, help='box embedding type')
    parser.add_argument("--gamma", default=0.3, type=float, help='box-vector distance parameter')

    return parser.parse_args()