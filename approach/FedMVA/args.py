import argparse
import torch


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--Epoch', type=int, default=50, help='number of rounds of training')
    parser.add_argument('--r', type=int, default=5, help='number of communication rounds')
    parser.add_argument('--client', type=int, default=3, help='number of total clients')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--Kp', type=int, default=0, help='number of personalized layers')
    parser.add_argument('--total', type=int, default=3, help='number of total layers')
    parser.add_argument('--optimizer', type=str, default='adam', help='type of optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--step_size', type=int, default=10, help='weight decay')
    parser.add_argument('--gamma', type=float, default=0.1, help='weight decay')
    parser.add_argument('--max_patience', type=int, default=5, help='patience')
    clients = ['VD' + str(i) for i in range(1,4)]
    parser.add_argument('--clients', default=clients)

    args = parser.parse_args()

    return args