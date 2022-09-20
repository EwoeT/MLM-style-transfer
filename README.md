# MLM-style-transfer

## Data
- Extracted sentences are available https://drive.google.com/drive/folders/1dT62-X2mDBjpe8yvd6flJizeF8mcbPpN?usp=sharing


## To train all mitigation components - mitigation_model, generate_neutral_latent_embedding_model, bias_class_discriminator:
```
!./train_mitigation_model.sh "binary_classification_data.csv" "bias_only.train.en" "bias_only.dev.en" "neutral_only.train.en" "neutral_only.dev.en"
```

# Requirements
torch==1.12.1 <br/>
transformers==4.21.2 <br/>
numpy<br/>



