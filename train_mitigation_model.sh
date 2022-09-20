classification_data_path=$1
bias_train_data_path=$2
bias_val_data_path=$3
neutral_train_data_path=$4
neutral_val_data_path=$5



python src/bias_class_discriminator.py \
-data_path $classification_data_path

python src/bias_classification_straight_through.py \

python src/latent_embedding_classifier.py \
-train_dataset_path "binary_bias_classification_train_dataset.pt" \
-val_dataset_path "binary_bias_classification_val_dataset.pt"

python src/generate_neutral_latent_representation.py \
-train_dataset_path $bias_train_data_path \
-val_dataset_path $bias_val_data_path \
-lambda1 0.5

python src/bias_mitigation_MLM.py \
-train_dataset_path $neutral_train_data_path \
-val_dataset_path $neutral_val_data_path \
-gamma 0.5
# -test_dataset_path "jigsaw_data/neutral_only.test.en"
