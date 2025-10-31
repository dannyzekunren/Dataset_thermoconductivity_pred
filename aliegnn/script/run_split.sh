cd $PBS_O_WORKDIR
module load cuda/11.8
module load nvhpc/25.3
module load oneapi

conda activate aliegnn

nvidia-smi

FILE_NAME="random"  # Default value

echo "Training with dataset: ${FILE_NAME}"

OUT_DIR="output_split_${FILE_NAME}_2_1"

DATA_ROOT="/home/wangzeyu/home/research/database/Dataset_thermoconductivity_pred/processed_splits"
CONFIG_FILE="config_random_split_2.json"
ID_PROP_FILE_TRAIN="${FILE_NAME}_train_data.csv"
ID_PROP_FILE_TEST="${FILE_NAME}_test_data.csv"

echo "Starting training with the following parameters:"
echo "DATA_ROOT: ${DATA_ROOT}"
echo "CONFIG_FILE: ${CONFIG_FILE}"
echo "OUTPUT_DIR: ${OUT_DIR}"
echo "TRAIN_FILE: ${ID_PROP_FILE_TRAIN}"
echo "TEST_FILE: ${ID_PROP_FILE_TEST}"

python train_folder_split_data.py \
    --root_dir $DATA_ROOT \
    --config_name $CONFIG_FILE \
    --output_dir $OUT_DIR \
    --id_prop_file_train $ID_PROP_FILE_TRAIN \
    --id_prop_file_test $ID_PROP_FILE_TEST \

echo "Job completed successfully."