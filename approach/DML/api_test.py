from server import FedPer
from args import args_parser

if __name__ == '__main__':

    args = args_parser()
    fedPer = FedPer(args)
    fedPer.server()

    fedPer.global_test()
