import os
import sys
import time
import numpy as np
import torch
from torchvision import transforms
import torch.utils.data as data
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu,SmoothingFunction

PRINT_FREQ = 500

def train(train_loader,encoder,decoder,criterion,
          optimizer,vocab_size,epoch,total_step,start_step=1,start_loss=0.0):
    """Train the encoder-decoder model for one epoch,save the check points evert 'PRINT_FREQ' steps and return the epoch's average train loss"""
    
    #Train mode
    encoder.train()
    decoder.train()
    
    #Track loss
    total_loss = start_loss
    
    #start time for each 100 steps
    start = time.time()
    
    for train_step in range(start_step,total_step+1):
        #sample a caption length and indices associted with it randomly
        indices = train_loader.dataset.get_indices()
        
        new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
        train_loader.batch_sampler.sampler = new_sampler
        
        #obtain the batch with sampled indices
        for batch in train_loader:
            images,captions = batch[0],batch[1]
            break
        
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()
        
        #Get the feature vector from CNN encoder and tokens from RNN decoder
        features = encoder(images)
        outputs = decoder(features,captions)
        
        #Batch loss calculation
        loss = criterion(outputs.view(-1,vocab_size),captions.view(-1))
        
        #Zero the gradient at start of every minibatches since backward() function accumaltes it
        optimizer.zero_grad()
        
        #Backpropagate loss to find weight gradients
        loss.backward()
        
        optimizer.step()
        
        total_loss+=loss.item()
        
        print("Epoch {},Step [{}/{}],{}s, Loss:{:.4f}\n".format(epoch,train_step,total_step,time.time()-start,loss.item()),end="")
        
        sys.stdout.flush()
        
        if train_step%PRINT_FREQ==0:
            print("Epoch {},Step [{}/{}],{}s, Loss:{:.4f}\n".format(epoch,train_step,total_step,time.time()-start,loss.item()))
            ckpt_file = os.path.join("./models","train-model-{}-{}.pkl".format(epoch,train_step))
            
            save_checkpoint(ckpt_file,encoder,decoder,optimizer,total_loss,epoch,train_step)
            start = time.time()
            
    return total_loss/total_step



def validate(val_loader,encoder,decoder,criterion,vocab,epoch,total_step,start_step=1,start_loss=0.0,start_bleu_4=0.0):
    """Validate the model for an epoch and return average validation loss and BLEU-4 score for each epoch"""
    
    #Evaluation mode
    encoder.eval()
    decoder.eval()
    
    smoothing_fn = SmoothingFunction()
    
    #Track val_loss and BLEU-4 score
    total_loss = start_loss
    total_bleu_4 = start_bleu_4
    
    start = time.time()
    
    with torch.no_grad():
        for val_step in range(start_step,total_step+1):
            indices = val_loader.dataset.get_indices()
            
            new_sampler=data.sampler.SubsetRandomSampler(indices=indices)
            val_loader.batch_sampler.sampler=new_sampler
            
            for batch in val_loader:
                images,captions = batch[0],batch[1]
                
            
            if torch.cuda.is_available():
                images = images.cuda()
                captions = captions.cuda()
            
            features = encoder(images)
            outputs  = decoder(features,captions)
            
            batch_bleu_4= 0.0
            
            for i in range(len(outputs)):
                pred_ids = list()
                for scores in outputs[i]:                  
                #Find the id of the token that has max probability for current step(greedy approach)
                    pred_ids.append(scores.argmax().item())
                #convert ids to tokens
                pred_word_list = word_list(pred_ids,vocab)
                gt_word_list = word_list(captions[i].numpy(),vocab)
                
                #Calculate BLEU-4 score
                batch_bleu_4+=sentence_bleu([gt_word_list],
                                           pred_word_list,
                                           smoothing_function=smoothing_fn.method1)
            total_bleu_4+=batch_bleu_4/len(outputs)
            
            loss = criterion(outputs.view(-1,len(vocab)),captions.view(-1))
            total_loss+=loss.item()
            
            print("Epoch {},Val Step [{}/{}],{}s, Loss: {:.4f}, VLEU-4: {:.4f} \n".format(epoch,val_step,total_step,time.time()-start,loss.item(),batch_bleu_4/len(outputs)))
            
            sys.stdout.flush()
            
            if val_step%PRINT_FREQ==0:
                print("Epoch {},Step [{}/{}],{}s, Loss:{:.4f}\n".format(epoch,val_step,total_step,time.time()-start,loss.item()))
                ckpt_file = os.path.join("./models","val-model-{}-{}.pkl".format(epoch,val_step))
            
                save_val_checkpoint(ckpt_file,encoder,decoder,total_loss,total_bleu4,epoch,val_step)
                start = time.time()
    
    return total_loss/total_step,total_bleu_4/total_step

def save_checkpoint(file,encoder,decoder,
                    optimizer,total_loss,epoch,train_step=1):
    
    torch.save({"encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "optimizer" : optimizer.state_dict(),
                "total_loss": total_loss,
                "epoch": epoch,
                "train_step": train_step,
               }, file)
    
def save_val_checkpoint(file,encoder,decoder,total_loss,
                        total_bleu4,epoch,val_step=1):
    
    
            torch.save({"encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "total_loss": total_loss,
                "total_bleu_4": total_bleu4,
                "epoch": epoch,
                "val_step": val_step,
               }, file)

                 
def save_epoch(filename, encoder, decoder, optimizer, train_losses, val_losses, 
               val_bleu, val_bleus, epoch):
    """Save at the end of an epoch. Save the model's weights along with the 
    entire history of train and validation losses and validation bleus up to 
    now, and the best Bleu-4."""
    torch.save({"encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "train_losses": train_losses,
                "val_losses": val_losses,
                "val_bleu": val_bleu,
                "val_bleus": val_bleus,
                "epoch": epoch
               }, filename)

def early_stopping(val_bleus, patience=3):
    """Check to stop training if  Bleu-4 scores remains constant for  'patience' 
    number of consecutive epochs."""
    
    if patience > len(val_bleus): # Min epeoch count should be 'patience'
        return False
    recent_bleu4_score = val_bleus[-patience:]
    
    
    if len(set(recent_bleu4_score)) == 1:
        return True
    max_bleu = max(val_bleus)
    if max_bleu in recent_bleu_score:
        #Check for convergence
        if max_bleu not in val_bleus[:len(val_bleus) - patience]:
            return False
        else:
            return True
    # If none of recent Bleu scores is greater than max_bleu, it has converged
    return True
            
def word_list(word_id_list, vocab):
    """
    Take word ids and vocabulary built from the dataset,
    return list of mapped words for the ids
    """
    word_list = []
    for i in range(len(word_idx_list)):
        vocab_id = word_idx_list[i]
        word = vocab.id2word[vocab_id]
        if word == vocab.end_seq:
            break
        if word != vocab.start_seq:
            word_list.append(word)
    return word_list

def clean_sentence(word_idx_list, vocab):
    """
    Take word ids and vocabulary built from the dataset,
    return sentence made up of that words
    """
    sentence = []
    for i in range(len(word_idx_list)):
        vocab_id = word_idx_list[i]
        word = vocab.id2word[vocab_id]
        if word == vocab.end_seq:
            break
        if word != vocab.start_seq:
            sentence.append(word)
    sentence = " ".join(sentence)
    return sentence

def get_prediction(data_loader=None, encoder=None, decoder=None, vocab=None,image_path=None):
    """
    Loop over images from the data loader or run on given image,
    predict the captions based on greedy search and beam search

    """
    if data_loader is not None:
        orig_image, image = next(iter(data_loader))
    elif image_path is not None:
        from PIL import Image

        transform_test = transforms.Compose([ 
        transforms.Resize(256),                          # smaller edge of image resized to 256
        transforms.CenterCrop(224),                      # get 224x224 crop from the center
        transforms.ToTensor(),                           # convert the PIL Image to a tensor
        transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                             (0.229, 0.224, 0.225))])
        
    


        PIL_image = Image.open(image_path)
        orig_image = np.array(PIL_image)
        image = transform_test(PIL_image)
    else:
        raise("You must pass a data_loader or an image path")

    plt.imshow(np.squeeze(orig_image))
    plt.title("Sample Image")
    plt.show()
    if torch.cuda.is_available():
        image = image.cuda()
    features = encoder(image).unsqueeze(1)
    print ("Caption without beam search:")
    output = decoder.greedy_search(features)
    sentence = clean_sentence(output, vocab)
    print (sentence)

    print ("Top captions using beam search:")
    outputs = decoder.beam_search(features)
    # Print maximum the top 3 predictions
    num_sents = min(len(outputs), 3)
    for output in outputs[:num_sents]:
        sentence = clean_sentence(output, vocab)
        print (sentence)      
        
        
        