import os
import random
import json
import numpy as np
import nltk
import torch
import torch.utils.data as data
from vocabulary import Vocabulary
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm


class CaptionDataset(data.Dataset):
    
    def __init__(self,transform,mode,batch_size,
                 threshold,sample_size,file,start_seq,end_seq,unk_word,
                annotations_file,load_vocab,image_dir):
        self.transform = transform
        self.mode = mode
        self.batch_size = batch_size
        self.vocab = Vocabulary(threshold,file,start_seq,end_seq,unk_word,annotations_file,load_vocab)
        self.image_dir = image_dir
        self.sample_size = sample_size
        
        if mode in ['train','val']:
            self.coco = COCO(annotations_file)
            self.ids = list(self.coco.anns.keys())
            print("IDS",len(self.ids))
            self.ids = self.ids[:self.sample_size]
            
            tokens = list()
            for idx in tqdm(np.arange(len(self.ids))):
                caption = str(self.coco.anns[self.ids[idx]]['caption']).lower()
                tokens.append(nltk.tokenize.word_tokenize(caption))
            
            self.caption_lengths=[len(token) for token in tokens]
            
        else:
            test_anns = json.load(open(annotations_file))
            self.paths = [item['file_name'] for item in test_anns['images']]
#             self.coco = COCO(annotations_file)
#             self.paths = os.listdir('val2014')
            
             
            
    def __getitem__(self,index):
        """Get the image and relevant captions"""
        if self.mode in ['train','val']:
            ann_id = self.ids[index]
            caption = self.coco.anns[ann_id]['caption']
            img_id =self.coco.anns[ann_id]['image_id']
            
            img_path = self.coco.loadImgs(img_id)[0]['file_name']
            
            #Transform the image to load into PyTorch
            image =Image.open(os.path.join(self.image_dir,img_path)).convert("RGB")
            image = self.transform(image)
            
            #Map captions to ids

            tokens = nltk.tokenize.word_tokenize(str(caption).lower())
            
            caption = []
            caption.append(self.vocab(self.vocab.start_seq))
            caption.extend([self.vocab(token) for token in tokens])
            caption.append(self.vocab(self.vocab.end_seq))
            caption = torch.Tensor(caption).long()
            
            return image,caption
        
        else:
            
            img_path = self.paths[index]
            
            test_img = Image.open(os.path.join(self.image_dir,img_path)).convert("RGB")
            orig_img = np.array(test_img)
            image = self.transform(test_img)
            
            return orig_img,image
        
    def get_indices(self):
        
        sel_len = np.random.choice(self.caption_lengths)
        all_indices = np.where([self.caption_lengths[i]==sel_len for i in np.arange(len(self.caption_lengths))])[0]
        indices = list(np.random.choice(all_indices,size=self.batch_size))
        
        return indices

    def __len__(self):
        if self.mode in ["train" ,"val"]:
            return len(self.ids)
        else:
            return len(self.paths)
        
def get_loader(transform,
              mode='train',
              batch_size=1,
              threshold=5,
              sample_size=30000,
              file = 'prebuilt_vocab.pkl',
              start_seq='<start>',
              end_seq='<end>',
              unk_word='<unk>',
              load_vocab=True,
              num_workers=0,
              cocoapi_loc='.',
              ):
    
    assert mode in ['train','val','test']
    if not(load_vocab):
        assert mode == 'train',"Vocabulary space can be built from scratch only when  mode = 'train"
        
        
    if mode =='train':
        if load_vocab:
            assert os.path.exists(file),"File does not exist"
        
        img_dir = os.path.join(cocoapi_loc,"train2014")
        annotations_file = os.path.join(cocoapi_loc,"annotations/captions_train2014.json")
    
    if mode =='val':
        assert load_vocab==True,"Vocab must be loaded from disk"
        assert os.path.exists(file),"File does not exist"
        
        img_dir = os.path.join(cocoapi_loc,"val2014")
        annotations_file = os.path.join(cocoapi_loc,"annotations/captions_val2014.json")
        
    if mode =='test':
        assert batch_size==1,"Batch size must be 1 in when testing"
        assert load_vocab==True,"Vocab must be loaded from disk"

        assert os.path.exists(file),"File does not exist"
         
        img_dir = os.path.join(cocoapi_loc,"test2014")
        annotations_file = os.path.join(cocoapi_loc,"annotations/image_info_test2014.json")
    
    dataset = CaptionDataset(transform=transform,
                            mode=mode,
                            batch_size=batch_size,
                            threshold=threshold,
                            sample_size=sample_size,
                            file=file,
                            start_seq=start_seq,
                            end_seq=end_seq,
                            unk_word=unk_word,
                            annotations_file=annotations_file,
                            load_vocab=load_vocab,
                            image_dir=img_dir)
    
    if mode=='train':
        #Select a random aption length and smple indices wth that length
        indices = dataset.get_indices()
        
        initial_sampler = data.sampler.SubsetRandomSampler(indices=indices)
        
        data_loader=data.DataLoader(dataset=dataset,
                                    num_workers=num_workers,
                                    batch_sampler=data.sampler.BatchSampler(sampler=initial_sampler,
                                                                            batch_size=dataset.batch_size,
                                                                            drop_last=False))
                                    
    else:
        data_loader = data.DataLoader(dataset=dataset,batch_size=dataset.batch_size,
                                      shuffle=True,
                                      num_workers=num_workers)
                                   
    return data_loader
                                    
        
        