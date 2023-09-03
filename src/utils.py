import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='ml1m', type=str, help='dataset')
    parser.add_argument("--gpu", default='0', type=str, help='gpu number')

    parser.add_argument("--batch_size", default=512, type=int, help='batch size')
    parser.add_argument("--epochs", default=200, type=int, help='number of epochs')
    parser.add_argument("--learning_rate", default=1e-3, type=float, help='learning rate')
    parser.add_argument("--dim", default=4, type=int, help='embedding dimension')
    parser.add_argument("--beta", default=1, type=float, help='beta for box smoothness')

    parser.add_argument("--K", default=30, type=int, help='quantization: K')
    parser.add_argument("--D", default=4, type=int, help='quantization: D')
    parser.add_argument("--tau", default=1.0, type=float, help='quantization: tau')
    parser.add_argument("--lmbda", default=0.1, type=float, help='lambda')

    parser.add_argument("--pos_instance", default=10, type=int, help='positive instance per set')
    parser.add_argument("--neg_instance", default=10, type=int, help='negative instance per set')

    return parser.parse_args()