#!/usr/bin/env python
# coding: utf-8

# In[53]:


import os
import numpy as np
import pandas as pd
import pickle
from PIL import Image
from tqdm.notebook import tqdm
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow.keras.utils import to_categorical, plot_model


# In[54]:


BASE_DIR = '/kaggle/input/flickr8k'
WORKING_DIR = '/kaggle/input/assets'


# In[55]:


model = VGG16()
model = Model(inputs = model.inputs, outputs = model.layers[-2].output)
model.summary()


# In[56]:


# Already save the features dictionary to assets folder

# features = {}
# img_dir = os.path.join(BASE_DIR, 'Images')
# for img_name in tqdm(os.listdir(img_dir)):
#     img_path = os.path.join(img_dir, img_name)
#     image = load_img(img_path, target_size = (224, 224))
#     image = img_to_array(image)
#     image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
#     image = preprocess_input(image)
#     feature = model.predict(image, verbose = 0)
#     img_id = img_name.split('.')[0]
#     features[img_id] = feature
    


# In[57]:


# pickle.dump(features, open(os.path.join(WORKING_DIR, 'features.pkl'), 'wb'))


# In[58]:


with open (os.path.join(WORKING_DIR, 'features.pkl'), 'rb') as f:
    features = pickle.load(f)


# In[59]:


with open(os.path.join(BASE_DIR, 'captions.txt'), 'r') as f:
    next(f)
    captions_doc = f.read()


# In[60]:


caption_mapping = {}
for line in tqdm(captions_doc.split('\n')):
    tokens = line.split(',')
    if len(line) < 2:
        continue
    image_id, caption = tokens[0], tokens[1:]
    image_id = image_id.split('.')[0]
    # converting list to string
    caption = " ".join(caption)
    if image_id not in caption_mapping:
        caption_mapping[image_id] = []
    caption_mapping[image_id].append(caption)


# In[61]:


len(caption_mapping)


# In[62]:


def clean(mapping):
    updated_mapping = {}
    for key, captions in mapping.items():
        updated_captions = []
        for caption in captions:
            processed_caption = caption.lower()
            processed_caption = processed_caption.replace('[^A-Za-z]', '')
            processed_caption = processed_caption.replace('\s+', ' ')
            processed_caption = 'startseq ' + ' '.join([word for word in processed_caption.split() if len(word) > 1]) + ' endseq'
            updated_captions.append(processed_caption)
        updated_mapping[key] = updated_captions
    return updated_mapping
            


# In[63]:


caption_mapping['1007320043_627395c3d8']


# In[64]:


updated_mapping = clean(caption_mapping)


# In[65]:


updated_mapping['1007320043_627395c3d8']


# In[66]:


all_captions = []
for key in updated_mapping.keys():
    for caption in updated_mapping[key]:
        all_captions.append(caption)


# In[67]:


len(all_captions)


# In[68]:


all_captions[:10]


# In[69]:


tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1
vocab_size


# In[70]:


max_length = max(len(caption.split()) for caption in all_captions)
max_length


# In[71]:


image_ids = list(updated_mapping.keys())
print(len(image_ids))
split = int(len(image_ids) * 0.9)
print('split -> ', split)
train = image_ids[:split]
test = image_ids[split:]


# In[72]:


def data_generator(data_keys, mapping, features, tokenizer, vocab_size, max_length, batch_size):
    X1, X2, y = list(), list(), list()
    n = 0
    while True:
        for key in data_keys:
            captions = mapping[key]
            n = n + 1
            for caption in captions:
                seq = tokenizer.texts_to_sequences([caption])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen = max_length)[0]
                    out_seq = to_categorical([out_seq], num_classes = vocab_size)[0]
                    X1.append(features[key][0])
                    X2.append(in_seq)
                    y.append(out_seq)
            if n == batch_size:
                X1, X2, y = np.array(X1), np.array(X2), np.array(y)
                yield [X1, X2], y
                X1, X2, y = list(), list(), list()
                n = 0
                


# In[73]:


l = 'startseq girl going into wooden building endseq'
seq = tokenizer.texts_to_sequences([l])[0]
print(seq)
for i in range(1, len(seq)):
    print('i -> ', i)
    in_seq, out_seq = seq[:i], seq[i]
    in_seq = pad_sequences([in_seq], maxlen = max_length)[0]
    out_seq = to_categorical([out_seq], num_classes = vocab_size)[0]


# In[74]:


# encoder model
# image feature layers
inputs1 = Input(shape=(4096,))
fe1 = Dropout(0.4)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)
# sequence feature layers
inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
se2 = Dropout(0.4)(se1)
se3 = LSTM(256)(se2)

# decoder model
decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)

model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam')

# plot the model
plot_model(model, show_shapes=True)


# In[75]:


# train the model
# Saved the trained model to assets folder


# epochs = 20
# batch_size = 32
# steps = len(train) // batch_size

# for i in range(epochs):
#     # create data generator
#     generator = data_generator(train, updated_mapping, features, tokenizer, vocab_size, max_length, batch_size)
#     # fit for one epoch
#     model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1)


# In[76]:


# Already saved the trained model to assets folder

# model.save(WORKING_DIR+'/best_model.h5')


# In[77]:


from tensorflow.keras.models import load_model
model = load_model(os.path.join(WORKING_DIR, 'best_model.h5'))
model.summary()


# In[78]:


def idx_to_word(idx, tokenizer):
    for word, index in tokenizer.word_index.items():
            if idx == index:
                return word
    return None


# In[79]:


def predict_caption(model, image, tokenizer, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen = max_length)
        y_hat = model.predict([image, sequence], verbose = 0)
        y_hat = np.argmax(y_hat)
        word = idx_to_word(y_hat, tokenizer)
        if word is None:
            break
        in_text = in_text + ' ' + word
        if word == 'endseq':
            break
    return in_text


# In[80]:


from nltk.translate.bleu_score import corpus_bleu
# validate with test data
actual, predicted = list(), list()

for key in tqdm(test):
    # get actual caption
    captions = updated_mapping[key]
    # predict the caption for image
    y_pred = predict_caption(model, features[key], tokenizer, max_length)
    # split into words
    actual_captions = [caption.split() for caption in captions]
    y_pred = y_pred.split()
    # append to the list
    actual.append(actual_captions)
    predicted.append(y_pred)
# calcuate BLEU score
print("BLEU-1: %f" % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
print("BLEU-2: %f" % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))


# In[81]:


import matplotlib.pyplot as plt
def generate_caption(image_name):
    image_id = image_name.split('.')[0]
    img_path = os.path.join(BASE_DIR, "Images", image_name)
    image = Image.open(img_path)
    captions = updated_mapping[image_id]
    print('---------------------Actual---------------------')
    for caption in captions:
        print(caption)
    # predict the caption
    y_pred = predict_caption(model, features[image_id], tokenizer, max_length)
    print('--------------------Predicted--------------------')
    print(y_pred)
    plt.imshow(image)


# In[82]:


generate_caption("1001773457_577c3a7d70.jpg")


# In[83]:


generate_caption("1002674143_1b742ab4b8.jpg")


# In[84]:


generate_caption("101669240_b2d3e7f17b.jpg")

