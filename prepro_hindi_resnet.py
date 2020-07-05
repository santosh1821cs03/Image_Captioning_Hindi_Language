from scipy import ndimage
from collections import Counter
import tensorflow as tf
import numpy as np
import pandas as pd
import hickle
import pickle
import imageio
import os
import json
from tensorflow.keras.applications.resnet import ResNet101
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet import preprocess_input, decode_predictions
from tensorflow.keras.models import Model
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
# config = tf.compat.v1.ConfigProto( device_count = {'GPU': 1 , 'CPU':4 } ) 
# sess = tf.Session(config=config) 
# keras.backend.set_session(sess)




def load_pickle(path):
    with open(path, 'rb') as f:
        file = pickle.load(f)
        print ('Loaded %s..' %path)
        return file  

def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        print ('Saved %s..' %path)

def _process_caption_data(caption_file, image_dir, max_length):
    with open(caption_file) as f:
        caption_data = json.load(f)

    
    id_to_filename = {image['id']: image['file_name'] for image in caption_data['images']}

    
    data = []
    for annotation in caption_data['annotations']:
        image_id = annotation['image_id']
        annotation['file_name'] = os.path.join(image_dir, id_to_filename[image_id])
        data += [annotation]
    
    
    caption_data = pd.DataFrame.from_dict(data)
    del caption_data['id']
    caption_data.sort_values(by='image_id', inplace=True)
    caption_data = caption_data.reset_index(drop=True)
    
    del_idx = []
    for i, caption in enumerate(caption_data['caption']):
        caption = caption.replace('.','').replace(',','').replace("'","").replace('"','')
        caption = caption.replace('&','and').replace('(','').replace(")","").replace('-',' ')
        caption = " ".join(caption.split())  
        
        caption_data.set_value(i, 'caption', caption.lower())
        if len(caption.split(" ")) > max_length:
            del_idx.append(i)
    print("The number of captions before deletion: %d"%len(caption_data))
    caption_data = caption_data.drop(caption_data.index[del_idx])
    caption_data = caption_data.reset_index(drop=True)
    print("The number of captions after deletion: %d" %len(caption_data))
    return caption_data


def _build_vocab(annotations, threshold=1):
    counter = Counter()
    max_len = 0
    for i, caption in enumerate(annotations['caption']):
        words = caption.split(' ') 
        for w in words:
            counter[w] +=1
        
        if len(caption.split(" ")) > max_len:
            max_len = len(caption.split(" "))

    vocab = [word for word in counter if counter[word] >= threshold]
    print ('Filtered %d words to %d words with word count threshold %d.' % (len(counter), len(vocab), threshold))

    word_to_idx = {u'<NULL>': 0, u'<START>': 1, u'<END>': 2}
    idx = 3
    for word in vocab:
        word_to_idx[word] = idx
        idx += 1
    print("Max length of caption: ", max_len)
    return word_to_idx


def _build_caption_vector(annotations, word_to_idx, max_length=15):
    n_examples = len(annotations)
    captions = np.ndarray((n_examples,max_length+2)).astype(np.int32)   

    for i, caption in enumerate(annotations['caption']):
        words = caption.split(" ") 
        cap_vec = []
        cap_vec.append(word_to_idx['<START>'])
        for word in words:
            if word in word_to_idx:
                cap_vec.append(word_to_idx[word])
        cap_vec.append(word_to_idx['<END>'])
        if(len(cap_vec)>(max_length + 2)):
          continue

        
        
        if len(cap_vec) < (max_length + 2):
            for j in range(max_length + 2 - len(cap_vec)):
                cap_vec.append(word_to_idx['<NULL>']) 
        
        captions[i, :] = np.asarray(cap_vec)
    print("Finished building caption vectors")
    return captions


def _build_file_names(annotations):
    image_file_names = []
    id_to_idx = {}
    idx = 0
    image_ids = annotations['image_id']
    file_names = annotations['file_name']
    for image_id, file_name in zip(image_ids, file_names):
        if not image_id in id_to_idx:
            id_to_idx[image_id] = idx
            image_file_names.append(file_name)
            idx += 1

    file_names = np.asarray(image_file_names)
    return file_names, id_to_idx


def _build_image_idxs(annotations, id_to_idx):
    image_idxs = np.ndarray(len(annotations), dtype=np.int32)
    image_ids = annotations['image_id']
    for i, image_id in enumerate(image_ids):
        image_idxs[i] = id_to_idx[image_id]
    return image_idxs


def main():
    
    batch_size = 100
    
    max_length = 15
    
    word_count_threshold = 1
     
    base_model = ResNet101(weights='imagenet')
    # base_model.summary()
    print(os.cpu_count())
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    # exit()

    #change layer from which output features is to be taken
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('conv5_block3_add').output)
    
    image_dir = 'image/%2014_resized/'

    # about 80000 images and 400000 captions for train dataset
    # train_dataset = _process_caption_data(caption_file='data/annotations/captions_train2014.json',
    #                                       image_dir='image/train2014_resized/',
    #                                       max_length=max_length)

    
    # val_dataset = _process_caption_data(caption_file='data/annotations/captions_val2014.json',
    #                                     image_dir='image/val2014_resized/',
    #                                     max_length=max_length)

    
    
    # val_cutoff = int(0.1 * len(val_dataset))
    # test_cutoff = int(0.2 * len(val_dataset))
    # print 'Finished processing caption data'
    
    # save_pickle(train_dataset, 'data/train/train.annotations.pkl')
    # save_pickle(val_dataset[:val_cutoff], 'data/val/val.annotations.pkl')
    # save_pickle(val_dataset[val_cutoff:test_cutoff].reset_index(drop=True), 'data/test/test.annotations.pkl')

    for split in ['train_hindi_corrected', 'val_hindi', 'test_hindi']:
        annotations = load_pickle('./data/%s/%s.annotations.pkl' % (split, split))

        if split == 'train_hindi':
            word_to_idx = _build_vocab(annotations=annotations, threshold=word_count_threshold)
            save_pickle(word_to_idx, './data/%s/word_to_idx.pkl' % split)
        
        captions = _build_caption_vector(annotations=annotations, word_to_idx=word_to_idx, max_length=max_length)
        save_pickle(captions, './data/%s/%s.captions.pkl' % (split, split))

        file_names, id_to_idx = _build_file_names(annotations)
        save_pickle(file_names, './data/%s/%s.file.names.pkl' % (split, split))

        image_idxs = _build_image_idxs(annotations, id_to_idx)
        save_pickle(image_idxs, './data/%s/%s.image.idxs.pkl' % (split, split))

        
        image_ids = {}
        feature_to_captions = {}
        i = -1
        for caption, image_id in zip(annotations['caption'], annotations['image_id']):
            if not image_id in image_ids:
                image_ids[image_id] = 0
                i += 1
                feature_to_captions[i] = []
            feature_to_captions[i].append(caption.lower() + ' .')
        save_pickle(feature_to_captions, './data/%s/%s.references.pkl' % (split, split))
        print ("Finished building %s caption dataset" %split)
    # for split in ['train_hindi', 'val_hindi', 'test_hindi']:
    #     anno_path = './data/%s/%s.annotations.pkl' % (split, split)
    #     save_path = './data/%s/%s.features_resnet101.hkl' % (split, split)
    #     annotations = load_pickle(anno_path)
    #     image_path = list(annotations['file_name'].unique())
    #     n_examples = len(image_path)

    #     all_feats = np.ndarray([n_examples, 196, 512], dtype=np.float32) #change here according to layer chosen

    #     for start, end in zip(range(0, n_examples, batch_size),
    #                           range(batch_size, n_examples + batch_size, batch_size)):
    #         image_batch_file = image_path[start:end]
    #         image_batch = np.array(list(map(lambda x: imageio.imread(x,as_gray=False,pilmode="RGB"), image_batch_file))).astype(np.float32)
    #         with tf.device('/gpu:0'):
    #             x = preprocess_input(image_batch)
    #             feats = model.predict_on_batch(x)
    #         all_feats[start:end, :] = feats.reshape(-1,196,512)
    #         print ("Processed %d %s features.." % (end, split))

    #     hickle.dump(all_feats, save_path)
    #     print ("Saved %s.." % (save_path))


if __name__ == "__main__":
    main()