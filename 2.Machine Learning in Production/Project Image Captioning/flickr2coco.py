import os
import random
import json
import numpy as np
from PIL import Image 
 



class JsonCreate():
    """docstring for JsonCreate"""
    def __init__(self,image_dir,captions_path,file_path,mode):
        super(JsonCreate, self).__init__()

        self.image_dir = image_dir
        self.file_path = file_path
        self.captions_path = captions_path
        self.captions_dict = None
        self.image_cap_dict = None
        self.img_dict = None
        self.annot_dict = None
        self.dump_dict = None
        self.mode = mode

        self.load_captions(self.captions_path)
        self.load_image_names(self.file_path)
        self.create_img_dict()
        self.create_annot_dict()
        self.dump_json()


     

    def load_captions(self,captions_path):

        captions_dict = dict()
        with open(captions_path)as f:
            cap_list = list()
            i = 1
            for line in (f):
                s = line.split('\t')

                cap_list.append(s[-1].rstrip())
                i+=1


                if(i%6==0):
                    img_id = s[0].split('.')[0]
                    captions_dict[img_id] = cap_list
                    cap_list = []
                    i = 1


                # for k,v in json.loads(line).items():
                #     captions_dict[k] = v

        self.captions_dict = captions_dict


    def load_image_names(self,file_path):

        image_cap_dict = dict()
        with open(file_path) as f:
            for line in f:
                img_name = line.rstrip()
                img_id = img_name.split('.')[0]
                image_cap_dict[img_id] = self.captions_dict[img_id]

        self.image_cap_dict = image_cap_dict

    def create_img_dict(self):

        #TO_DO : add license details
        img_list= list()
        img_dict = {'images':dict()}

        for i,(k,v) in enumerate(self.image_cap_dict.items()):
            im = Image.open(os.path.join(self.image_dir,k+'.jpg'))
            h,w,c = np.array(im).shape
            temp_dict = {
                        "id":k,
                        "file_name":k+".jpg",
                        "height":h,
                        "width":w
                        }

            img_list.append(temp_dict)

        img_dict['images'] = img_list

        self.img_dict = img_dict


    def create_annot_dict(self):

        annot_dict = {'annotations':dict()}
        annot_list = list()
         

        
        for i,(_id,captions) in enumerate(self.image_cap_dict.items()): 
            for caption in captions:
                temp_dict = {
                            "image_id":_id,
                            "id" : _id.split('_')[0],
                            "caption":caption
                            }
                 
                annot_list.append(temp_dict)

        annot_dict['annotations'] = annot_list
        self.annot_dict = annot_dict

    def dump_json(self):

        dump_dict = {
                    'info':{"description":'FLICKR8K Dataset'},
                            'images':self.img_dict['images'],
                            'licenses':"To be added",
                            'annotations':self.annot_dict['annotations']
                            

                    }
        self.dump_dict = dump_dict
        with open('Flickr8k_{}.json'.format(self.mode),'w')as f:

            json.dump(dump_dict,f)


# if __name__ == '__main__':
#     img_dir = '../Flickr8k/Flicker8k_Dataset'
#     file_path = '../Flickr8k/Flickr8k_text/Flickr_8k.testImages.txt'
#     cpations_path = ..
#     json_create = JsonCreate()









        





