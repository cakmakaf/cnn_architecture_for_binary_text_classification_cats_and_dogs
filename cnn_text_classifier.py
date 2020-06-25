#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 22:42:37 2020

@author: ahmetcakmak
"""


from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D,Conv1D,MaxPool1D
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import Model
from sklearn.model_selection import train_test_split
from cnn_classifier_helper import load_data
import numpy as np

print('Loading data')
x, y,xtest,testsentences, vocabulary, vocabulary_inv = load_data()

# Randomly divide the dataset into two part as train and validation sets. (X_test = validation set)
X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=42)

# if the classes are no balanced, we upsampled lower class with adding random data from the same class.
if True:
    nc=y_train[:,1].sum()
    nd=y_train[:,0].sum()

    if nc>nd:
        I=np.where(y_train[:,0]==1)[0]
        J=np.random.choice(I,nc-nd)      
    elif nd>nc:
        I=np.where(y_train[:,1]==1)[0]
        J=np.random.choice(I,nd-nc)

    X_train=np.vstack((X_train,X_train[J,:]))
    y_train=np.vstack((y_train,y_train[J,:]))


# Used 1-hot encoding for 10 words. Each word is a vector out of 2529 probability space.
# We need to induce it to the feature vectors or in other words we need to do a word embedding.
# There are multiple werd embedding methods such as GloVe, word2vec... In here, I build my own
# choosing dimension of 32 for embedding layer. It tels us the length of the feature vector for each word.
sequence_length = x.shape[1] 
vocabulary_size = len(vocabulary_inv) 
embedding_dim = 32 
# The size of the sliding windows at each convolution.
filter_sizes = [3,4,5]
num_filters = 64
drop = 0.5

epochs = 50
batch_size = 30


# To convert each word to 32x1 vector we need 2529x32 parameters.
# For the embedding data wi will have 32x10 = 320 embedded layers. 
# 10 dim --embed--> 10x32 dim
print("Creating Model...")
inputs = Input(shape=(sequence_length,), dtype='int32')
embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=sequence_length)(inputs)


# Then convolute 10x32 layers from 3 different layers. 
# The first convolution will have inputs-outputs as 10x32 dim --conv1--> 8x64 dim and,
conv_0 = Conv1D(num_filters, kernel_size=filter_sizes[0], padding='valid', kernel_initializer='normal', activation='relu')(embedding)
# The second convolution will have inputs-outputs as 10x32 dim --conv1--> 7x64 dim and,
conv_1 = Conv1D(num_filters, kernel_size=filter_sizes[1], padding='valid', kernel_initializer='normal', activation='relu')(embedding)
# The third convolution will have inputs-outputs as 10x32 dim --conv1--> 6x64 dim and,
conv_2 = Conv1D(num_filters, kernel_size=filter_sizes[2], padding='valid', kernel_initializer='normal', activation='relu')(embedding)


# and max-pooling as 8x64__maxpool1-->64
maxpool_0 = MaxPool1D(pool_size=sequence_length - filter_sizes[0] + 1,  padding='valid')(conv_0)
# and max-pooling as 7x64__maxpool1-->64
maxpool_1 = MaxPool1D(pool_size=sequence_length - filter_sizes[1] + 1,  padding='valid')(conv_1)
# and max-pooling as 8x64__maxpool1-->64
maxpool_2 = MaxPool1D(pool_size=sequence_length - filter_sizes[2] + 1,  padding='valid')(conv_2)



# Then we concatenate them as [64, 64, 64]--concat--> 64x3
concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
# Flatten the tensor as 64x3--flatten-->192
flatten = Flatten()(concatenated_tensor)
# We used 0.5 drop out and as 192--Dropout(0.5)-->192. That is we drop the half of it randomly
# and refill it by zeros. 
dropout = Dropout(drop)(flatten)
# We pass it to the dense layer as 192--Dense-->2
output = Dense(units=2, activation='softmax')(dropout)


# Here, build a model as 10--model-->2
model = Model(inputs=inputs, outputs=output)

model.summary()

# The checkpoint allows us to find the best validation accuracy and print the results
# to the 'weights.best.hdf5' file. 
checkpoint = ModelCheckpoint('weights.best.hdf5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
# One of the optimiztion algorithms to train the NN. The hyper parameters coul be switched.  
adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

# Fids the model that makes the cross-entropy minimum
model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
print("Traning Model...")
# this fits the model 
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[checkpoint], validation_data=(X_test, y_test))  

model.load_weights("weights.best.hdf5")
model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

# Evaluate the accuracy of our trained model
yhat=model.predict(xtest)

# Print the class score for each sentence classified 
for i in range(0,len(testsentences)):
    print(testsentences[i])
    print('Dog:'+str(yhat[i,0])+' Cat:'+str(yhat[i,1]))

a=1