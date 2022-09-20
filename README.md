# MLM-style-transfer

## 1. Data prep
- We use code from https://github.com/kanekomasahiro/context-debias/blob/main/script/preprocess.sh to extract senteces containing attribute abd stereotype words. <br/>
- Extracted sentences are available https://drive.google.com/drive/folders/1dT62-X2mDBjpe8yvd6flJizeF8mcbPpN?usp=sharing
## 2. Data tokenization
- Tokenize sentences; positional indices of attribute/target words are kept for embedding extraction.  Use attribute for training the detector and stereotypes to test for bias only<br/>
- Args: <br/>
-model_type: type of model "albert-large" or "bert-large <br/>
-data_types: **"attributes"** for generating tokens for attributes and **"stereotypes"** for generating tokens for stereotypes. <br/>
-data_path: path to data <br/>
-save_tokenized_data_path: path to save tokenized data <br/>
-female_list_path: path to female list (attributes or stereotypes) <br/>
-male_list_path: path to male attributes (attributes or stereotypes) <br/>
-all_attributes_and_names_path: path to file containing female and male attributes and names to exclude from sentences containing stereotypes (optional). Removes gender from the context of stereotypes
-sequence_length: max number of tokens to generate per sentence (optional, default: 4) <br/>

- To train all mitigation components - mitigation_model, generate_neutral_latent_embedding_model, bias_class_discriminator:
```
!./train_mitigation_model.sh "binary_classification_data.csv" "bias_only.train.en" "bias_only.dev.en" "neutral_only.train.en" "neutral_only.dev.en"
```

# Requirements
torch==1.12.1 <br/>
transformers==4.21.2 <br/>
numpy<br/>



