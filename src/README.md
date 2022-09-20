# MLM-style-transfer

Bert model is adapted from huggingface https://huggingface.co/transformers/model_doc/bert.html
Bert fine-tuning codes are adapted from:https://mccormickml.com/2019/07/22/BERT-fine-tuning/

## bias_class_discriminator.ipynb: To train classifier for bias detection
__Params:__

- data_path", required=True)
- sequence_length", type=int, default=100)
- seed_value", type=int, default=42)
- save_model_path", default="one_hot_classifier.pth")
- epochs", type=int, default=4)
- batch_size", type=int, default=32)

## bias_classification_straight_through.ipynb: To trian classifier with straight through technique
__Params:__

- train_dataset_path", default="binary_bias_classification_train_dataset.pt")
- val_dataset_path", default="binary_bias_classification_val_dataset.pt")
- sequence_length", type=int, default=100)
- seed_value", type=int, default=42)
- save_model_path", default="one_hot_classifier.pth")
- epochs", type=int, default=4)
- batch_size", type=int, default=32)

## latent_embedding_classifier.ipynb: Trains classifier to detect if latent encoding is biased or neutral (in the case of bias mitigation) / male or female (in the case of gender obfuscation)

- train_dataset_path", required=True)
- val_dataset_path", required=True)
- sequence_length", type=int, default=100)
- seed_value", type=int, default=42)
- save_model_path", default="latent_embedding_discriminator.pth")
- epochs", type=int, default=4)
- batch_size", type=int, default=32)

## generate_neutral_latent_representation.ipynb: Generates disentangled (neutral) latent representation
- train_dataset_path", required=True)
- val_dataset_path", required=True)
- sequence_length", type=int, default=100)
- seed_value", type=int, default=42)
- save_model_path", default="generate_neutral_latent_embedding_model.pth" )
- epochs", type=int, default=4)
- batch_size", type=int, default=32)
- lambda1", type=float, default=0.5)

## bias_mitigation_MLM.ipynb: Main style transfer code
- train_dataset_path", required=True)
- val_dataset_path", required=True)
- test_dataset_path", required=True)
- sequence_length", type=int, default=100)
- gamma", type=float, default=0.5)
- seed_value", type=int, default=42)
- save_model_path", default="mitigation_model.pth")
- epochs", type=int, default=4)
- batch_size", type=int, default=32)

# Requirements
transformers==4.10.0
