import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='path to model',
                    required=True)
parser.add_argument('--output', type=str, help='output model path',
                    required=True)
parser.add_argument('--log_file', type=str, help='log file',
                    required=True)
parser.add_argument('--gpu', type=int, help='on which gpu will be compute',
                    default=0)
parser.add_argument('--batch', type=int, help='training batch size',
                    default=32)
parser.add_argument('--img_size', type=int, help='img height/width',
                    default=64)

train_args = parser.parse_args()
