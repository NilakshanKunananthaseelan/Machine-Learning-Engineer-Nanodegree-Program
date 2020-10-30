import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F


class ResNetEncoder(nn.Module):
    
    def __init__(self,embedding_size):
        """Load the pretrained ResNet-50 and replace top linear layer"""
        super(ResNetEncoder,self).__init__()
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]
        #Create a sequential models upto top fc layer add a custom fc layer compatible with embedding size of decoder RNN
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features,embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size,momentum=0.01)
        self.init_weights()
    
    def init_weights(self):
        self.embed.weight.data.normal_(0.0,0.01)
        self.embed.bias.data.fill_(0)
    
    def forward(self,images):
        """Extract feature vector from input image"""
        with torch.no_grad(): 
            features = self.resnet(images)
        features = features.view(features.size(0),-1)
        features = self.embed(features)
        features = self.bn(features)
        return features

class InceptionEncoder(nn.Module):
    
    def __init__(self,embedding_size):
        super(InceptionEncoder,self).__init__()
        
        inception = models.inception_v3(pretrained=True)
        modules = list(inception.children())[:-1]
        
        self.inception = nn.Sequential(*modules)
        self.embed = nn.Linear(inception.fc.in_features,embedding_size)
        
        self.bn = nn.BatchNorm1d(embedding_size,momentum=0.01)
        self.init_weights()
        
    def init_weights(self):
        self.embed.weight.data.normal_(0.0,0.01)
        self.embed.bias.data.fill_(0)
        
    def forward(self,images):
        "Extract feature vector from input image"
        with torch.no_grad():
            features = self.inception(images)
        features = features.view(features.size(0),-1)
        features = self.embed(features)
        features = self.bn(features)
        return features
    
class RNNDecoder(nn.Module):
    def __init__(self,
                 embedding_size,
                 hidden_size,
                 vocab_size,
                 num_layers=1):
        """Build LSTM layers to decode"""
        super(RNNDecoder,self).__init__()
        self.embed = nn.Embedding(vocab_size,embedding_size)
        self.lstm  = nn.LSTM(embedding_size,hidden_size,num_layers,batch_first=True)
        self.linear = nn.Linear(hidden_size,vocab_size)
        
    def forward(self,features,captions):
        """Decode image feature vector to generate captions"""
        captions = captions[:,:-1]
        embeddings = self.embed(captions)
        inputs = torch.cat((features.unsqueeze(1),embeddings),1)
        hiddens,_ = self.lstm(inputs)
        outputs = self.linear(hiddens)
        
        return outputs
    
    def greedy_search(self,inputs,states=None,max_len=20):
        """Take an image feature vector and return predicted token ids of max_len.In this greedy approach we generate tokens by passing image feature vector to a LSTM layer.The token with maximum probability is taken as the predicted one,then it is used as input to next layer and this process continued till generated sequence is of length,max_len"""
        
        ids_list = list()
        for i in range(max_len):
            hiddens,states = self.lstm(inputs,states)
            outputs = self.linear(hiddens.squeeze(1))
            #Get the most likely integer to represent the token
            
            predicted = outputs.argmax(1)
            ids_list.append(predicted.item())
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)
        return ids_list
    
    def beam_search(self,inputs,states=None,max_len=20,beam_width=5):
        """Take an image feature vector,and return top 'beam_width' sized predicted tokens at each LSTM layer"""
        
        idx_seq = [[[],0.0,inputs,states]]
        for i in range(max_len):
            #Store all top candiates of token at each step
            top_candidates = list() 
             
            #predict next word id for each of top tokens previously generated
            for idx in idx_seq:
                hiddens,states = self.lstm(idx[2],idx[3])
                outputs = self.linear(hiddens.squeeze(1))
                
                #Since softmax probabilities are very small,converting to log will avoid potential FP underflow
                log_probs = F.log_softmax(outputs,-1)
                top_log_probs,top_idx = log_probs.topk(beam_width,1)
                top_idx = top_idx.squeeze(0)
                
                #generate a new set of top sentences 
                
                for j in range(beam_width):
                    next_idx,log_prob = idx[0][:],idx[1]
                    next_idx.append(top_idx[j].item())
                    log_prob += top_log_probs[0][j].item()
                    
                    inputs = self.embed(top_idx[j].unsqueeze(0)).unsqueeze(0)
                    
                    top_candidates.append([next_idx,log_prob,inputs,states])
            
            sorted_prob = sorted(top_candidates,key=lambda pair:pair[1],reverse=True)
            idx_seq = sorted_prob[:beam_width]
            
        return [idx[0] for idx in idx_seq]
            
                    
                    
            
    
    
    
        
        
        
        