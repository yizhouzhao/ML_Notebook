import os, json
import numpy as np
import h5py
import pandas as pd
import tensorflow as tf
import urllib.request, urllib.error, urllib.parse, os, tempfile
import uuid
"""

This library is input pipeline for coco captioning
dataset

"""

DATA_DIR = '/home/karen/workspace/data/coco_captioning'
HEIGHT = 224
WIDTH = 224

def image_from_url(url, fname):
    """
    Read an image from a URL. Returns a numpy array with the pixel data.
    We write the image to a temporary file then read it back. Kinda gross.
    """
    try:
        f = urllib.request.urlopen(url)
        with open(fname, 'wb') as ff:
            ff.write(f.read())
        return fname
    except urllib.error.URLError as e:
        print('URL Error: ', e.reason, url)
    # except:
    #     print("Other error:")

def url_to_image(features, caption):
    fname = features["filename"]
    img = tf.read_file(fname)
    img = tf.image.decode_jpeg(img, 3)
    img = tf.image.resize_images(img, (HEIGHT, WIDTH))
    features["img"] = img
    return (features, caption)

class CocoCaptionData:
    def __init__(self, data_dir):
        self.caption_folder = os.path.join(data_dir, 'annotations')
        # self.train_feat_pca_file = os.path.join(data_dir, 'train2014_vgg16_fc7_pca.h5')
        # self.train_feat_file = os.path.join(data_dir, 'train2014_vgg16_fc7.h5')

        # self.val_feat_pca_file = os.path.join(data_dir, 'val2014_vgg16_fc7_pca.h5')
        # self.val_feat_file = os.path.join(data_dir, 'val2014_vgg16_fc7.h5')

        # self.dict_file = os.path.join(data_dir, 'coco2014_vocab.json')
        self.train_url_file = os.path.join(data_dir, 'train2017')
        self.val_url_file = os.path.join(data_dir, 'val2017')

        # self.dict_data = self.get_vocab()

    def get_captions(self, is_training):
        if not is_training:
            img_path = self.val_url_file
            path = os.path.join(self.caption_folder, "captions_val2017.json")
        else:
            img_path = self.train_url_file
            path = os.path.join(self.caption_folder, "captions_train2017.json")
        caption_data = {}
        image_data = {}
        with open(path, 'r') as f:
            json_data = json.load(f)
            ids = []
            image_ids = []
            caption = []
            for l in json_data["annotations"]:
                ids.append(l["id"])
                image_ids.append(l["image_id"])
                caption.append(l["caption"])
            caption_data["id"] = np.array(ids)
            caption_data["image_id"] = np.array(image_ids)
            caption_data["caption"] = np.array(caption)


            image_data = {l["id"]:os.path.join(img_path, l["file_name"]) for l in json_data["images"]}
            # image_data["image_id"] = np.array([l["image_id"] for l in json_data["images"]])

        return caption_data, image_data



    # def get_vocab(self):
    #     with open(self.dict_file, 'r') as f:
    #         dict_data = json.load(f)
    #         return dict_data

    def sample_data(self):
        caption_data, image_data = self.get_captions(True)
        # features = self.get_image_url(True)
        return caption_data, image_data

        
    # def get_image_url(self, is_training):
    #     if is_training:
    #         url_files = self.train_url_file
    #     else:
    #         url_files = self.val_url_file
    #     with open(url_files, 'r') as f:
    #         return np.asarray([line.strip() for line in f])

    def coco_input(self, data_dir, is_training, pca_features, use_feature, epochs, buffer_size, batch_size):
        # Get Captioning Data
        caption_data, image_data = self.get_captions(is_training)
        # if use_feature:
        #     features = self.get_features(pca_features, is_training)
        # else:
        # features = self.get_image_url(is_training)


        captions = caption_data["caption"]
        image_id = caption_data["image_id"]
        caption_id = caption_data["id"]

        def input_fn():
            file_name = np.array([np.array(image_data[i]) for i in image_id])
            ds = tf.data.Dataset.from_tensor_slices((dict(filename=file_name), captions))
            # if not use_feature:
            ds = ds.map(url_to_image)
            if is_training:
                ds = ds.shuffle(buffer_size=buffer_size)
                ds = ds.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
                ds = ds.repeat(epochs)
            else:
                ds = ds.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
            
            ds = ds.prefetch(1)
            iterator = ds.make_one_shot_iterator()
            output = iterator.get_next()

            return output

        return input_fn
    
    @staticmethod
    def get_vocab(caption_data):
        vocab_data = set()
        for c in caption_data:
            for v in c.split(" "):
                vocab_data.add(v)
        return list(vocab_data)

############################################################
################  Preprocessed Feature Map      ############
############################################################


def decode_captions(captions, idx_to_word):
    singleton = False
    if captions.ndim == 1:
        singleton = True
        captions = captions[None]
    decoded = []
    N, T = captions.shape
    for i in range(N):
        words = []
        for t in range(T):
            word = idx_to_word[captions[i, t]]
            if word != '<NULL>':
                words.append(word)
            if word == '<END>':
                break
        decoded.append(' '.join(words))
    if singleton:
        decoded = decoded[0]
    return decoded


def sample_coco_minibatch(data, batch_size=100, split='train'):
    split_size = data['%s_captions' % split].shape[0]
    mask = np.random.choice(split_size, batch_size)
    captions = data['%s_captions' % split][mask]
    image_idxs = data['%s_image_idxs' % split][mask]
    image_features = data['%s_features' % split][image_idxs]
    urls = data['%s_urls' % split][image_idxs]
    return captions, image_features, urls


############################################################
################  Image Preprocessing      ##################
############################################################
