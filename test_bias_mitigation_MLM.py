import tensorflow as tf
import numpy as np
import torch
import random
import pandas as pd
import os, sys
import time
import datetime
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from sentence_transformers import util
import argparse


# random.seed(42)
torch.manual_seed(42) 
if torch.cuda.is_available():
  device = torch.device("cuda")
  print('There are %d GPU(s) available.' % torch.cuda.device_count())
  print('We will use the GPU:', torch.cuda.get_device_name(0))

else:
  print('No GPU available, using the CPU instead.')
  device = torch.device("cpu")
    

    
    

parser = argparse.ArgumentParser()
parser.add_argument("-test_dataset_path", required=True)
parser.add_argument("-mitigation_model_path", default="src/mitigation_model.pth")
parser.add_argument("-generate_neutral_latent_embedding_model_path", default="src/generate_neutral_latent_embedding_model.pth")
parser.add_argument("-bias_class_discriminator_path", default="src/bias_class_discriminator.pth")
parser.add_argument("-sequence_length", type=int, default=300)
parser.add_argument("-seed_value", type=int, default=42)
parser.add_argument("-output_path", default="output.txt")
parser.add_argument("-threshold_value", type=float, default=0.1)
# parser.add_argument("-batch_size", type=int, default=32)
args = parser.parse_args()
config = vars(args)
  
test_dataset_path = args.test_dataset_path
mitigation_model_path = args.mitigation_model_path
generate_neutral_latent_embedding_model_path = args.generate_neutral_latent_embedding_model_path
bias_class_discriminator_path = args.bias_class_discriminator_path
# test_dataset_path = args.test_dataset_path
# mitigation_model_path = args.mitigation_model_path
# epochs = args.epochs
# batch_size = args.batch_size
threshold_value = args.threshold_value
seed_val = args.seed_value
output_path = args.output_path
seq_len = args.sequence_length

    
X_test = pd.read_csv(test_dataset_path, header=None)
X_test = X_test[0].values

model = torch.load(mitigation_model_path)
model.sbert.load_state_dict(torch.load(generate_neutral_latent_embedding_model_path).bias_mitigation_model.state_dict())
bias_detector = torch.load(bias_class_discriminator_path)

from bert_model import SBERT
similarity_model = SBERT.from_pretrained("bert-base-uncased")

########################### load functions ###########################
# from transformers import BertTokenizer, AdamW, BertConfig, BertForPreTraining

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
vocabs = tokenizer.get_vocab()
# seq_len = 300
def tokenizze(data):
    

    # Load the BERT tokenizer.

    # Training Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    attention_masks = []
    target_labels = []

    # For every sentence...
    for k, sent in enumerate(data):
        encoded_dict = tokenizer.encode_plus(
                            str(sent),                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = seq_len,           # Pad & truncate all sentences.
                            truncation=True,
                            padding = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                       )
        
        
        tokens_ids=encoded_dict['input_ids'][0].cpu().numpy().copy()
#         target_output = tokens.copy()

        tokens_tensor = torch.tensor([tokens_ids])
        input_ids.append(tokens_tensor)
        attention_masks.append(encoded_dict['attention_mask'])
        target_labels.append(encoded_dict['input_ids'])
        
        
    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    target_labels = torch.cat(target_labels, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    return input_ids, target_labels, attention_masks

#  Lime for explainability
import sklearn
from lime import lime_text
from lime.lime_text import LimeTextExplainer
label_names = [0, 1]
explainer = LimeTextExplainer(kernel_width = 25, class_names=label_names)


# senten = X_test[0]


from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
def predictor(texts):
    result = []
    for text in texts:
        encoded_dict = tokenizer.encode_plus(
                                str(text),                      # Sentence to encode.
                                add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                                max_length = seq_len,           # Pad & truncate all sentences.
                                truncation=True,
                                padding = True,
                                return_attention_mask = True,   # Construct attn. masks.
                                return_tensors = 'pt',     # Return pytorch tensors.
                           )

         # Add the encoded sentence to the list.    
        inputIds = encoded_dict['input_ids']

        # And its attention mask (simply differentiates padding from non-padding).
        attentionMask = encoded_dict['attention_mask']
        labell = [1]
        labell  = torch.tensor(labell)
        bias_detector.eval()

        inputIds = inputIds.to(device)
        inputMask = attentionMask.to(device)
        labell = labell.to(device)
        with torch.no_grad():
            (t_loss, t_logits) = bias_detector(inputIds, 
                                           token_type_ids=None, 
                                           attention_mask=inputMask,
                                           labels=labell
                                              )



        t_logits = np.array(t_logits.cpu().numpy())

        t_logits = [list(i) for i in t_logits]
        t_logits = list(t_logits)

        predictions = tf.nn.softmax(t_logits)
        predictions = np.array(predictions)
        predictions = [list(i) for i in predictions]
        predictions = np.array(list(predictions[0]))
        result.append(predictions)
    return np.array(result)


def predict_instance(text):
    encoded_dict = tokenizer.encode_plus(
                            str(text),                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = seq_len,           # Pad & truncate all sentences.
                            truncation=True,
                            padding = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                       )

     # Add the encoded sentence to the list.    
    inputIds = encoded_dict['input_ids']

    # And its attention mask (simply differentiates padding from non-padding).
    attentionMask = encoded_dict['attention_mask']
    labell = [1]
    labell  = torch.tensor(labell)
    bias_detector.eval()

    inputIds = inputIds.to(device)
    # inputIdss = inp6.to(device)
    inputMask = attentionMask.to(device)
    # inputMask = att6.to(device)
    labell = labell.to(device)
    with torch.no_grad():
        (t_loss, t_logits) = bias_detector(inputIds, 
                                       token_type_ids=None, 
                                       attention_mask=inputMask,
                                       labels=labell
                                       )


    return t_logits



import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
StopWords = stopwords.words("english")

def replace_word(text, findWord, replaceWord):
    text_tokens = word_tokenize(text)
    return ' '.join(replaceWord if word.lower() == findWord.lower() else word for word in text_tokens)

def mask_polarized_tokens(text):
    original_text = text
    text_tokens = word_tokenize(original_text)
    num_samples = 1000
    exp = explainer.explain_instance(text, predictor, num_features=10, num_samples=num_samples)
    words = exp.as_list()
    words = sorted(words, reverse=True, key=lambda x: x[1])
    scores = [i[1] for i in words]
    tokens_list = [i[0] for i in words]
    important_words =[]
    count = 0
    for key, score in enumerate(scores):
        if score>threshold_value:
            important_words.append(tokens_list[key])
        if len(important_words) == 0:
            important_words.append(tokens_list[0])
    text_tokens = ' '.join(["[MASK]" if word in important_words else word for word in text_tokens])
    return text_tokens, words

def fix_wordpiece(wordpiece_tokens):
    for key, i in reversed(list(enumerate(wordpiece_tokens))):
        if i.startswith("##"):
#             print(i)
#             print("".join([wordpiece_tokens[key-1], wordpiece_tokens[key][2:]]))
            wordpiece_tokens[key-1] = "".join([wordpiece_tokens[key-1], wordpiece_tokens[key][2:]])
            del wordpiece_tokens[key]
    return wordpiece_tokens

def mitigate(TEXT, source_TEXT):
    inss = tokenizze([TEXT])
    source_input = tokenizze([source_TEXT])

    input_ids = inss[0].to(device)
    attention_mask = inss[2].to(device)
    source_labels = source_input[0].to(device)
    labels_attention_mask = source_input[2].to(device)
    outputs = model(input_ids, 
                                     token_type_ids=None, 
                                     attention_mask=attention_mask,
                                     labels=source_labels,
                                     labels_attention_mask=labels_attention_mask, 
                   )
    vocabs = tokenizer.get_vocab()
    vocabs = dict((v,k) for k,v in vocabs.items())
 
    import numpy as np
    ids_list = []
    new_text_list = []
    pred_score = outputs[1][0].cpu().detach().numpy()
    for outt in pred_score:
        pred_flat = np.argmax(outt).flatten()
        ids_list.append(np.argmax(outt))
        new_text_list.append(vocabs[pred_flat[0]])
    new_text_list = new_text_list[1:-1]
    fix_wordpiece(new_text_list)
    ids_list = torch.LongTensor([ids_list])
    resultwords = new_text_list
    new_text = ' '.join(resultwords)
    attention_mask = attention_mask.cpu().detach()
    return new_text, ids_list, attention_mask
    


########################## Test with test data ####################################



import time
import datetime

# def format_time(elapsed):
#     '''
#     Takes a time in seconds and returns a string hh:mm:ss
#     '''
#     # Round to the nearest second.
#     elapsed_rounded = int(round((elapsed)))
    
#     # Format as hh:mm:ss
#     return str(datetime.timedelta(seconds=elapsed_rounded))



import numpy as np

import random
import numpy as np
from tqdm.notebook import tqdm
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

# This training code is based on the `run_glue.py` script here:
# https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128
# Set the seed value all over the place to make this reproducible.
# seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)




   # ========================================
#               Testing
# ========================================
# After the completion of each training epoch, measure our performance on
# our validation set.

print("")
print("Running test...")
total_bias_score = 0
total_similarity_score = 0
count = 0
start_time = time.time()

# Put the model in evaluation mode--the dropout layers behave differently
# during evaluation.



# # Tracking variables 
# total_eval_masked_accuracy = 0
# total_eval_loss = 0
# nb_eval_steps = 0


# Evaluate data for one epoch
for ind, text in enumerate(X_test[610:1000]):
   
    #print(text, "\n")
    text_embedding = tokenizze([text])
    text_logits = predict_instance(text)[0].cpu().numpy()
    text_score = np.argmax(text_logits)
#     print(text)
    masked_TEXT, words = mask_polarized_tokens(text)
#     print(masked_TEXT, "\n")
#     try:
    count = count + 1
    debiased_text, debiased_text_embedding, masked_attention_mask = mitigate(masked_TEXT, text)

#     print(debiased_text_embedding)
    #     calculate bias
    bias_logits = predict_instance(debiased_text)[0].cpu().numpy()
    bias_score = np.argmax(bias_logits)
    total_bias_score = total_bias_score + bias_score
    

   
    
#     calculate similarity
    emb1 = similarity_model(text_embedding[0], token_type_ids=None, attention_mask=text_embedding[2])
    emb2 = similarity_model(debiased_text_embedding, token_type_ids=None, attention_mask=masked_attention_mask)
#     print(emb1.shape)
    similarity = util.pytorch_cos_sim(emb1, emb2).item()
    total_similarity_score = total_similarity_score + similarity


    
    print("\n ############# original text ##################### \n", text, "############# score: ", text_score,  "############# similarity: ", 1.000)
    print("\n# \n", words)
    print("\n ############### masked text ################### \n", masked_TEXT )
    print("\n ############## debiased text #################### \n", debiased_text, "############# score: ", bias_score , "############# similarity: ", similarity)
  
    
    
    with open(output_path, "a") as dd_text:
            dd_text.write(f"{debiased_text}\n")


    

