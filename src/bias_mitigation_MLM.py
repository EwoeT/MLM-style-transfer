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
parser.add_argument("-train_dataset_path", required=True)
parser.add_argument("-val_dataset_path", required=True)
# parser.add_argument("-test_dataset_path", required=True)
parser.add_argument("-sequence_length", type=int, default=100)
parser.add_argument("-gamma", type=float, default=0.5)
parser.add_argument("-seed_value", type=int, default=42)
parser.add_argument("-save_model_path", default="mitigation_model.pth")
parser.add_argument("-epochs", type=int, default=4)
parser.add_argument("-batch_size", type=int, default=32)
args = parser.parse_args()
config = vars(args)
  
train_dataset_path = args.train_dataset_path
val_dataset_path = args.val_dataset_path
# test_dataset_path = args.test_dataset_path
gamma = args.gamma,
epochs = args.epochs
batch_size = args.batch_size
seed_val = args.seed_value
save_model_path = args.save_model_path
seq_len = args.sequence_length
 



########################### Load dataset ###########################
X_train = pd.read_csv(train_dataset_path, header=None)
X_val = pd.read_csv(val_dataset_path, header=None)
# X_test = pd.read_csv(test_dataset_path, header=None)
X_train = X_train[0].values
X_val = X_val[0].values
# X_test = X_test[0].values


########################### tokenize dataset ###########################
# from transformers import BertTokenizer, AdamW, BertConfig, BertForPreTraining

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
vocabs = tokenizer.get_vocab()
seq_len = 30
def tokenize(data):
    

    # Load the BERT tokenizer.
    print('Loading BERT tokenizer...')

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
                            pad_to_max_length = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                       )
        
        
        tokens_ids=encoded_dict['input_ids'][0].cpu().numpy().copy()
#         target_output = tokens.copy()

        for i, token in enumerate(tokens_ids):
            mask_candidates = random.sample(range(0, len(tokens_ids)), round(len(tokens_ids)*0.25))
            # mask selected tokens
            if i in mask_candidates and (tokens_ids[i] > 1013):
            #  with  80% probability, mask token
                prob = random.random()
                if prob < 0.8:
                    tokens_ids[i] = tokenizer.mask_token_id


        tokens_tensor = torch.tensor([tokens_ids])
        input_ids.append(tokens_tensor)
        attention_masks.append(encoded_dict['attention_mask'])
        target_labels.append(encoded_dict['input_ids'])
        
        
    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    target_labels = torch.cat(target_labels, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    return input_ids, target_labels, attention_masks




train_input_ids, train_target_labels, train_attention_masks = tokenize(X_train)
val_input_ids, val_target_labels, val_attention_masks = tokenize(X_val)
# test_input_ids, test_target_labels, test_attention_masks = tokenize(X_test)

print('Original: ', X_train[0])
print('Token IDs:', train_input_ids[0])
print('Token IDs:', train_target_labels[0])



from torch.utils.data import TensorDataset

train_dataset = TensorDataset(train_input_ids, train_target_labels, train_attention_masks)
val_dataset = TensorDataset(val_input_ids, val_target_labels, val_attention_masks)
# test_dataset = TensorDataset(test_input_ids, test_target_labels, test_attention_masks)

print('{:>5,} training samples'.format(len(train_dataset)))
print('{:>5,} validation samples'.format(len(val_dataset)))
# print('{:>5,} test samples'.format(len(test_dataset)))


########################### Load model ###########################
from bert_model import BertForMaskedLM

model = BertForMaskedLM.from_pretrained("bert-base-uncased")

# Tell pytorch to run this model on the GPU.
model.cuda()
model.sbert.load_state_dict(torch.load("generate_neutral_latent_embedding_model.pth").bias_mitigation_model.state_dict())


########################### optimization, training, and validation functions ###########################
from transformers.optimization import AdamW
# Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
optimizer = AdamW(model.parameters(),
                  lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )


from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

# The DataLoader needs to know our batch size for training, so we specify it 
# here. For fine-tuning BERT on a specific task, the authors recommend a batch 
# size of 16 or 32.
batch_size = 32

# Create the DataLoaders for our training and validation sets.
# We'll take training samples in random order. 
train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )

# For validation the order doesn't matter, so we'll just read them sequentially.
validation_dataloader = DataLoader(
            val_dataset, # The validation samples.
            sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )

# # For validation the order doesn't matter, so we'll just read them sequentially.
# test_dataloader = DataLoader(
#             test_dataset, # The validation samples.
#             sampler = SequentialSampler(test_dataset), # Pull out batches sequentially.
#             batch_size = batch_size # Evaluate with this batch size.
#         )


from transformers.optimization import get_linear_schedule_with_warmup

# Number of training epochs. The BERT authors recommend between 2 and 4. 
# We chose to run for 4, but we'll see later that this may be over-fitting the
# training data.
epochs = 4

# Total number of training steps is [number of batches] x [number of epochs]. 
# (Note that this is not the same as the number of training samples).
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)



import time
import datetime

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))



import numpy as np

# Function to calculate the accuracy of our predictions vs labels
def flat_masked_accuracy(preds, labels):
#     pred_score = preds
    logits_argmax = np.array([np.argmax(l, axis=1) for l in preds])
    pred_flat = logits_argmax.flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)



import random
import numpy as np
from tqdm.notebook import tqdm
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

# This training code is based on the `run_glue.py` script here:
# https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128
def train(model, optimizer, losslogger, start_epoch, epochs, run_id, train_dataloader, validation_dataloader, checkpoint_name):
    # Set the seed value all over the place to make this reproducible.
    seed_val = 42

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # We'll store a number of quantities such as training and validation loss, 
    # validation accuracy, and timings.
    training_stats = []

    # Measure the total training time for the whole run.
    total_t0 = time.time()

    # For each epoch...
    t = tqdm(range(start_epoch,epochs))

    for epoch_i in t:

    # ========================================
    #               Training
    # ========================================

    # Perform one full pass over the training set.

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0
        total_bias_loss = 0
        total_mlm_loss = 0

        # Put the model into training mode. Don't be mislead--the call to 
        # `train` just changes the *mode*, it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):

            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            # Unpack this training batch from our dataloader. 
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the 
            # `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids 
            #   [1]: attention masks
            #   [2]: labels 
            b_input_ids = batch[0].to(device)
            b_target_labels = batch[1].to(device)
            b_input_mask = batch[2].to(device)

            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because 
            # accumulating the gradients is "convenient while training RNNs". 
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()        

            # Perform a forward pass (evaluate the model on this training batch).
            # The documentation for this `model` function is here: 
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # It returns different numbers of parameters depending on what arguments
            # arge given and what flags are set. For our useage here, it returns
            # the loss (because we provided labels) and the "logits"--the model
            # outputs prior to activation.
#             loss, logits, class_logits = model(b_input_ids,
            loss, masked_logits, class_logits = model(b_input_ids, 
                                 token_type_ids=None,
                                 gamma=gamma,
                                 attention_mask=b_input_mask, 
                                 labels=b_target_labels,
                                 labels_attention_mask=b_input_mask
                                              )

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value 
            # from the tensor.
            
#             print(bias_logits)

            total_train_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

            
        # Calculate the average mlm loss over all of the batches.
        avg_mlm_loss = total_mlm_loss / len(train_dataloader) 
        
        # Calculate the average bias loss over all of the batches.
        avg_bias_loss = total_bias_loss / len(train_dataloader) 
        
        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)            

        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        # set in logs
        df = pd.DataFrame()
        df['chackpoint_name'] = pd.Series(checkpoint_name)
        df['epoch'] = pd.Series(epoch_i)
        df['Loss'] = pd.Series(loss.data.item())
        df['run'] = run_id
        losslogger = losslogger.append(df)
        
        print("")
        print("  Average mlm loss: {0:.2f}".format(avg_mlm_loss))
        print("  Average bias loss: {0:.2f}".format(avg_bias_loss))
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))
        state = {'epoch': epoch_i + 1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'losslogger': losslogger, }
        torch.save(state, f'{checkpoint_name}')

        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        print("")
        print("Running Validation...")

        t0 = time.time()

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()

        
        # Tracking variables 
        total_eval_masked_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0


        # Evaluate data for one epoch
        for batch in validation_dataloader:

            # Unpack this training batch from our dataloader. 
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using 
            # the `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids 
            #   [1]: attention masks
            #   [2]: labels 
            b_input_ids = batch[0].to(device)
            b_target_labels = batch[1].to(device)
            b_input_mask = batch[2].to(device)

            # Tell pytorch not to bother with constructing the compute graph during
            # the forward pass, since this is only needed for backprop (training).
            with torch.no_grad():        

                # Forward pass, calculate logit predictions.
                # token_type_ids is the same as the "segment ids", which 
                # differentiates sentence 1 and 2 in 2-sentence tasks.
                # The documentation for this `model` function is here: 
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                # Get the "logits" output by the model. The "logits" are the output
                # values prior to applying an activation function like the softmax.
#                 (loss, logits, class_logits) = model(b_input_ids,
                (loss, masked_logits, class_logits) = model(b_input_ids,
                                 token_type_ids=None,
                                 gamma=gamma,
                                 attention_mask=b_input_mask, 
                                 labels=b_target_labels,
                                 labels_attention_mask=b_input_mask
                                                    )
    
    #             calculate MLM evaluation
            # Accumulate the validation loss.
            total_eval_loss += loss.item()

            # Move logits and labels to CPU
            masked_logitss = masked_logits.detach().cpu().numpy()
            b_target_labelss = b_target_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences, and
            # accumulate it over all batches.
            total_eval_masked_accuracy += flat_masked_accuracy(masked_logitss, b_target_labelss)
            
#            

        # Report the final accuracy for this validation run.
        avg_val_masked_accuracy = total_eval_masked_accuracy / len(validation_dataloader)
        print("  Masked Accuracy: {0:.2f}".format(avg_val_masked_accuracy))
        
         # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(validation_dataloader)


        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)

        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Masked Accur.': avg_val_masked_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )
   
    
    
def load_checkpoint(model, optimizer, losslogger, filename):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        losslogger = checkpoint['losslogger']
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, start_epoch, losslogger          
          
          
########################### Start training with checkpoint ###########################
# first training epoch
# Start
start_epoch = 0

# Logger
losslogger = pd.DataFrame()

# Checkpoint name
checkpoint_name = 'checkpoint.pth.tar'

train(model, optimizer, losslogger, start_epoch, epochs, 0, train_dataloader, validation_dataloader, checkpoint_name)
time.sleep(8)
          
torch.save(model, save_model_path)
    
