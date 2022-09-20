# MLM-style-transfer

Bert model is adapted from huggingface https://huggingface.co/transformers/model_doc/bert.html
Bert fine-tuning codes are adapted from:https://mccormickml.com/2019/07/22/BERT-fine-tuning/

## bias_class_discriminator.ipynb: To train classifier for bias detection <br/>- Params:

- train_dataset_path", required=True)
- val_dataset_path", required=True)
- sequence_length", type=int, default=100)
- seed_value", type=int, default=42)
- save_model_path", default="one_hot_classifier.pth")
- epochs", type=int, default=4)
- batch_size", type=int, default=32)

## 2_bias_classification_straight_through.ipynb: To trian classifier with straight through technique
## 3_latent_embedding_classifier.ipynb: Trains classifier to detect if latent encoding is biased or neutral (in the case of bias mitigation) / male or female (in the case of gender obfuscation)
## 4_generate_neutral_latent_representation.ipynb: Generates disentangled (neutral) latent representation
## 5_bias_mitigation_MLM.ipynb: Main style transfer code
## 6_TEST_bias_mitigation_MLM.ipynb: Code for evaluation

# Requirements
transformers==4.10.0
