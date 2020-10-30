import os

import numpy as np
import nltk
nltk.download('punkt')
import json
import pickle
from pycocotools.coco import COCO
from collections import Counter

class Vocabulary(object):
    
    def __init__(self,
                threshold,
                file='./prebuilt_vocab.pkl',
                start_seq='<start>',
                end_seq='<end>',
                unk_word='<unk>',
                anns_file='annotations/captions_train2014.json',
                # data='coco',
                load_vocab=False):
        """Initialize the Vocabulary class"""
    
        self.threshold = threshold
        self.file = file
        self.start_seq = start_seq
        self.end_seq = end_seq
        self.unk_word = unk_word
        self.annotations_file = anns_file
        self.load_vocab = load_vocab
        # self.data = data
        self.get_vocab()
        
    def get_vocab(self):
        """Load pre-built vocab file or build vocab dict from scratch"""
        if os.path.exists(self.file) and self.load_vocab:
            with open(self.file,'rb') as f:
                vocab = pickle.load(f)
                self.word2id = vocab.word2id
                self.id2word = vocab.id2word
                print('Loaded pre-built vocab file')
        else:
            self.build_vocab()
            
            with open(self.file,'wb') as f:
                pickle.dump(self,f)
    
    def build_vocab(self):
        """Map tokens to integers and integers to tokens"""
        self.init_vocab_build()
        self.add_new_word(self.start_seq)
        self.add_new_word(self.end_seq)
        self.add_new_word(self.unk_word)
        self.add_captions()
    
    def init_vocab_build(self):
        self.word2id=dict()
        self.id2word=dict()
        self.idx = 0
        
    def add_new_word(self,word):
        "Add tokens to vocabulary for mapping"
        if word not in self.word2id:
            self.word2id[word] = self.idx
            self.id2word[self.idx] = word
            self.idx+=1
    
    def add_captions(self):
        """Add all the tokens from captions to the vocabulary space which meet threshold requirement"""
        counter = Counter()
        # if self.data=='coco':
        coco = COCO(self.annotations_file)
        
        ids = coco.anns.keys()
        for i,_id in enumerate(ids):
            
            caption = str(coco.anns[_id]['caption'])
            tokens = nltk.tokenize.word_tokenize(caption.lower())
            counter.update(tokens)
        # elif self.data='flickr8k':
        #     with open(annotations_file)as f:
        #         for line in f:
        #             caption = line.split('\t')[-1].rstrip()
        #             tokens  = tokens = nltk.tokenize.word_tokenize(caption.lower())
        #             counter.update(tokens)




        # else:
        #     raise("use only data='coco' or data='flickr8k")
            
        print("Tokenizing Captions Ended")
        
        words = [word for word,_count in counter.items() if _count>=self.threshold ]
        for word in words:
            self.add_new_word(word)
    
    def __call__(self,word):
        if word not in self.word2id:
            return self.word2id[self.unk_word]
        return self.word2id[word]
    
    def __len__(self):
        return len(self.word2id)
            
        