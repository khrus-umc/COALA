'''
Main script that orchestrates running the COALA model on given input arguments.
'''
import argparse
from utils import *
from CRLM.helpers.eval import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CRLM segmentation')
    parser.add_argument('--task_id', type=str, default='Task001_CRLM')
    parser.add_argument('--nnunet_dir', type=str, default="model")
    parser.add_argument('--input_dir', type=str,
                        default="data/in/")
    parser.add_argument('--output_dir', type=str,
                        default="data/out/")

    args = parser.parse_args()

    print("Making lowres prediction...")
    run_inference(args.nnunet_dir, args.input_dir, args.output_dir, args.task_id, 'lowres')
    print("Successfully made lowres prediction...")

    print("Making fullres prediction...")
    run_inference(args.nnunet_dir, args.input_dir, args.output_dir, args.task_id, 'cascade_fullres')
    print("Successfully made fullres prediction...")

    print("Extracting volume...")
    extract_volume(args.input_dir, args.output_dir)
    print("Succesfully extracted volume...")
