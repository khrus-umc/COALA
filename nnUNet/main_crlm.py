import argparse
from utils import *
from CRLM.helpers.eval import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CRLM segmentation')
    parser.add_argument('--task_id', type=str, default='Task001_CRLM')
    parser.add_argument('--nnunet_dir', type=str, default="your_folder\\nnUNet\\")
    parser.add_argument('--input_dir', type=str, default="your_folder\\nnUNet\\prediction_input\\")
    parser.add_argument('--output_dir', type=str, default="your_folder\\nnUNet\\prediction_output\\")
    parser.add_argument('--data_dir', type=str, default="your_folder\\nnUNet\\CRLM\\")

    args = parser.parse_args()

    print("Renaming files...")
    rename_files(args.input_dir, '_0000')
    print("Successfully renamed files...")

    print("Making lowres prediction...")
    run_inference(args.nnunet_dir, args.input_dir, args.output_dir, args.task_id, '3d_lowres')
    print("Successfully made lowres prediction...")

    print("Making fullres prediction...")
    run_inference(args.nnunet_dir, args.input_dir, args.output_dir, args.task_id, '3d_cascade_fullres')

    print("Successfully made fullres prediction...")

    print("Extracting volume...")
    move(args.output_dir, os.path.join(args.data_dir, 'ai_segmentations'))
    extract_volume(args.input_dir, args.output_dir)
    print("Succesfully extracted volume...")
