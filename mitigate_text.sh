test_dataset_path=$1
mitigation_model_path=$2
generate_neutral_latent_embedding_model_path=$3
bias_class_discriminator_path=$4



python src/test_bias_mitigation_MLM.py \
-test_dataset_path $test_dataset_path \
-mitigation_model_path $mitigation_model_path \
-generate_neutral_latent_embedding_model_path $generate_neutral_latent_embedding_model_path \
-bias_class_discriminator_path $bias_class_discriminator_path