#!/bin/bash

nvidia-smi

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Runtime Parameters
MODEL_NAME=""
DATASET_NAME=""

# Default Argument Values
BATCH_SIZE=32
SEQ_LEN=1024
VOCAB_SIZE=50257

NUM_ITERATIONS=1000
OUTPUT_TENSOR_NAMES="last_hidden_state"

BYPASS_ARGUMENTS=""

# Loop through arguments and process them
for arg in "$@"
do
    case $arg in
        --model_name=*)
        MODEL_NAME="${arg#*=}"
        shift # Remove --model_name from processing
        ;;
        --dataset_name=*)
        DATASET_NAME="${arg#*=}"
        shift # Remove --dataset_name= from processing
        ;;
        --batch_size=*)
        BATCH_SIZE="${arg#*=}"
        shift # Remove --batch_size= from processing
        ;;
        --sequence_length=*)
        SEQ_LEN="${arg#*=}"
        shift # Remove --sequence_length= from processing
        ;;
        --vocab_size=*)
        VOCAB_SIZE="${arg#*=}"
        shift # Remove --vocab_size= from processing
        ;;
        --num_iterations=*)
        NUM_ITERATIONS="${arg#*=}"
        shift # Remove --num_iterations= from processing
        ;;
        --output_tensors_name=*)
        OUTPUT_TENSOR_NAMES="${arg#*=}"
        shift # Remove --output_tensors_name= from processing
        ;;
        --data_dir=*)
        DATA_DIR="${arg#*=}"
        shift # Remove --data_dir= from processing
        ;;
        --input_saved_model_dir=*)
        MODEL_DIR="${arg#*=}"
        shift # Remove --input_saved_model_dir= from processing
        ;;
        --tokenizer_model_dir=*)
        TOKENIZER_DIR="${arg#*=}"
        shift # Remove --tokenizer_model_dir= from processing
        ;;
        --total_max_samples=*)
        shift # Remove --total_max_samples= from processing
        ;;
        *)
        BYPASS_ARGUMENTS=" ${BYPASS_ARGUMENTS} ${arg}"
        ;;
    esac
done

echo -e "\n********************************************************************"
echo "[*] MODEL_NAME: ${MODEL_NAME}"
echo "[*] DATASET_NAME: ${DATASET_NAME}"
echo ""
echo "[*] DATA_DIR: ${DATA_DIR}"
echo "[*] MODEL_DIR: ${MODEL_DIR}"
echo "[*] TOKENIZER_DIR: ${TOKENIZER_DIR}"
echo ""
# Custom gpt2 Task Flags
echo "[*] SEQ_LEN: ${SEQ_LEN}"
echo "[*] VOCAB_SIZE: ${VOCAB_SIZE}"
echo "[*] OUTPUT_TENSOR_NAMES: ${OUTPUT_TENSOR_NAMES}"
echo ""
echo "[*] BYPASS_ARGUMENTS: $(echo \"${BYPASS_ARGUMENTS}\" | tr -s ' ')"

echo -e "********************************************************************\n"

DATA_DIR="${DATA_DIR}/${DATASET_NAME}"
MODEL_DIR="${MODEL_DIR}/${MODEL_NAME}/model"
TOKENIZER_DIR="${TOKENIZER_DIR}/${MODEL_NAME}/tokenizer"

if [[ ! -d ${DATA_DIR} ]]; then
    echo "ERROR: \`--data_dir=/path/to/directory\` does not exist. [Received: \`${DATA_DIR}\`]"
    exit 1
fi

if [[ ! -d ${MODEL_DIR} ]]; then
    echo "ERROR: \`--input_saved_model_dir=/path/to/directory\` does not exist. [Received: \`${MODEL_DIR}\`]"
    exit 1
fi

if [[ ! -d ${TOKENIZER_DIR} ]]; then
    echo "ERROR: \`--tokenizer_model_dir=/path/to/directory\` does not exist. [Received: \`${TOKENIZER_DIR}\`]"
    exit 1
fi

# Dataset Directory

python ${BASE_DIR}/infer.py \
    --data_dir=${DATA_DIR} \
    --calib_data_dir=${DATA_DIR} \
    --input_saved_model_dir=${MODEL_DIR} \
    --tokenizer_model_dir=${TOKENIZER_DIR} \
    --batch_size=${BATCH_SIZE} \
    --sequence_length=${SEQ_LEN} \
    --vocab_size=${VOCAB_SIZE} \
    --output_tensors_name=${OUTPUT_TENSOR_NAMES} \
    `# The following is set because we will be running synthetic benchmarks` \
    --total_max_samples=1 \
    --use_synthetic_data  \
    --num_iterations=${NUM_ITERATIONS} \
    ${@}