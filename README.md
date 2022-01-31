# MLM-style-transfer

Bert model is adapted from huggingface https://huggingface.co/transformers/model_doc/bert.html
Bert fine-tuning codes are adapted from:https://mccormickml.com/2019/07/22/BERT-fine-tuning/

- 1_bias_class_discriminator.ipynb: to train classifier
- 2_bias_classification_straight_through.ipynb: To trianclassifier with straight through technique
-3_latent_embedding_classifier.ipynb: trains classifier to detect if latent encoding is biased or neutral (in the case of bias mitigation) / male or female (in the case of gender opbfuscation)
- 4_generate_neutral_latent_representation.ipynb: Generates disentangled (neutral) latent representation
- 5_bias_mitigation_MLM.ipynb: Main style transfer code
- 6_TEST_bias_mitigation_MLM.ipynb: Code for evaluation

# Requirements
transformers==4.1.0
