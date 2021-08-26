from __future__ import print_function
import argparse
import os
import nmdr

if __name__ == '__main__':
    parser = argparse.ArgumentParser('CNN Exercise.')
    parser.add_argument('--mode',
                        type=int, default=1,
                        help='Select mode between 1-5.')
    parser.add_argument('--learning_rate',
                        type=float, default=0.001,
                        help='Initial learning rate.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=30,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--batch_size',
                        type=int, default=16,
                        help='Batch size. Must divide evenly into the dataset sizes.')
    parser.add_argument('--log_dir',
                        type=str,
                        default='logs',
                        help='Directory to put logging.')
    parser.add_argument('--debug',
                        type=bool,
                        default=False,
                        help='Enable debug mode.')
    parser.add_argument('--name',
                        type=str,
                        default='model',
                        help='Set model name')
    parser.add_argument('--num_classes',
                        type=str,
                        default='4',
                        help='Set number of classes')
    
    FLAGS = None
    FLAGS, unparsed = parser.parse_known_args()
    
    run_main(FLAGS)

def run_main(FLAGS):
    print(FLAGS)
