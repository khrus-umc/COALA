#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=0-04:00:00

module load 2021
module load Python/3.9.5-GCCcore-10.3.0
source nnenv/bin/activate

NNUNET_DIR='your_path'

INPUT_DIR="${NNUNET_DIR}prediction_input\\"
OUTPUT_DIR="${NNUNET_DIR}prediction_output\\"

python main_sectra.py --nnunet_dir "$NNUNET_DIR" --input_dir "$INPUT_DIR" --output_dir "$OUTPUT_DIR"
