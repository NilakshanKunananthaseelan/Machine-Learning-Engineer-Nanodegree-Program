import os

import torch
import os
from torchvision import transforms
import numpy as np
from coco_dataloader import get_loader
from model import ResNetEncoder,RNNDecoder
from PIL import Image
from pycocotools.coco import COCO
from vocabulary import Vocabulary
import streamlit as st

@st.cache(allow_output_mutation=True)

def get_prediction(image):

	#image = Image.
	transform_test = transforms.Compose([ 
	transforms.Resize(256),                          # smaller edge of image resized to 256
	transforms.CenterCrop(224),                      # get 224x224 crop from the center
	transforms.ToTensor(),                           # convert the PIL Image to a tensor
	transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                         (0.229, 0.224, 0.225))])
	
	
	orig_img = np.array(image)
	test_img = transform_test(image)
	sample_vocab = Vocabulary(threshold=5,load_vocab=True,anns_file = "captions_train2014.json")
	vocab_size = len(sample_vocab)

	#Model
	
	checkpoint = torch.load('train-model-1-9900.pkl')

	# Specify values for embed_size and hidden_size - we use the same values as in training step
	embed_size = 256
	hidden_size = 512

	 

	# Initialize the encoder and decoder, and set each to inference mode
	encoder = ResNetEncoder(embed_size)
	encoder.eval()
	decoder = RNNDecoder(embed_size, hidden_size, vocab_size)
	decoder.eval()

	# Load the pre-trained weights
	encoder.load_state_dict(checkpoint['encoder'])
	decoder.load_state_dict(checkpoint['decoder'])

	# Move models to GPU if CUDA is available.
	if torch.cuda.is_available():
	    encoder.cuda()
	    decoder.cuda()
	    image = image.cuda()
	test_img = test_img.unsqueeze(0)
	 
	features = encoder(test_img).unsqueeze(1)
	output = decoder.greedy_search(features)

	cleaned_pred = []
	
	for i in range(len(output)):
		vocab_id = output[i]
		word = sample_vocab.id2word[vocab_id]
		if word ==sample_vocab.end_seq:
			break
		if word != sample_vocab.start_seq:
			cleaned_pred.append(word)
	caption = " ".join(cleaned_pred)
	
	return caption
	
img = Image.open('test.jpg')
#img.show()

#print(get_prediction(img))	
	    
	
