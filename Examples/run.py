from __future__ import print_function
import argparse
import os
import nmdr

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Nano Mtrl Dev Research')
    parser.add_argument('--mode',
                        type=int, default=1,
                        help='Select mode between 1-5.')
    
    FLAGS = None
    FLAGS, unparsed = parser.parse_known_args()
    
    run_main(FLAGS)

def run_main(FLAGS):
    print(FLAGS)
