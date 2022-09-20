# MLM-style-transfer

## Data
- Extracted sentences are available https://drive.google.com/drive/folders/1seAMJn3Fh8ZXhDpiW_ih7HDBlqHhIcnu?usp=sharing


## To train all mitigation components using default parameters: 
- mitigation_model
- generate_neutral_latent_embedding_model
- bias_class_discriminator <br/>
TO define parameters, wee  __src__.

```
./train_mitigation_model.sh "binary_classification_data.csv" "bias_only.train.en" "bias_only.dev.en" "neutral_only.train.en" "neutral_only.dev.en"
```
## To mitigate text:
```
python f_TEST_bias_mitigation_MLM.py \
-test_dataset_path "bias_only.test.en" \
-mitigation_model_path "mitigation_model.pth" \
-generate_neutral_latent_embedding_model_path "generate_neutral_latent_embedding_model.pth" \
-bias_class_discriminator_path "bias_class_discriminator.pth"
```

# Requirements
torch==1.12.1 <br/>
transformers==4.21.2 <br/>
lime==0.2.0.1 <br/>
sentence_transformers==2.2.2 <br/>
nltk==3.7 <br/>
numpy <br/>

