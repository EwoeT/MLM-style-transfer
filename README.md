# MLM-style-transfer
Bert model is adapted from huggingface https://huggingface.co/transformers/model_doc/bert.html
Bert fine-tuning codes are adapted from:https://mccormickml.com/2019/07/22/BERT-fine-tuning/

## Data
- Extracted sentences are available https://drive.google.com/drive/folders/1seAMJn3Fh8ZXhDpiW_ih7HDBlqHhIcnu?usp=sharing


## To train all mitigation components using default parameters (To define parameters, see  _src_): 
- mitigation_model
- generate_neutral_latent_embedding_model
- bias_class_discriminator <br/>

```
./train_mitigation_model.sh "binary_classification_data.csv" "bias_only.train.en" "bias_only.dev.en" "neutral_only.train.en" "neutral_only.dev.en"
```
## To mitigate text:
```
python test_bias_mitigation_MLM.py \
-test_dataset_path "bias_only.test.en" \
-mitigation_model_path "src/mitigation_model.pth" \
-generate_neutral_latent_embedding_model_path "src/generate_neutral_latent_embedding_model.pth" \
-bias_class_discriminator_path "src/bias_class_discriminator.pth"
```
__Params:__
- test_dataset_path", required=True)
- mitigation_model_path", default="src/mitigation_model.pth")
- generate_neutral_latent_embedding_model_path", default="src/generate_neutral_latent_embedding_model.pth")
- bias_class_discriminator_path", default="src/bias_class_discriminator.pth")
- sequence_length", type=int, default=300)
- seed_value", type=int, default=42)
- threshold_value", type=float, default=0.1)
- output_path", default="output.txt")


# Requirements
torch==1.12.1 <br/>
transformers==4.21.2 <br/>
lime==0.2.0.1 <br/>
sentence_transformers==2.2.2 <br/>
nltk==3.7 <br/>
numpy <br/>

