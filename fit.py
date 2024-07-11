import os
import numpy as np
import keras
import pickle


from keras.models import model_from_json
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import Callback

import tensorflow as tf
tf.config.list_physical_devices('GPU')

class MyGenerator(keras.utils.Sequence):
    def __init__(self, data_dir, labels_dir, batch_size=1):
        self.data_dir = data_dir
        self.labels_dir = labels_dir
        self.batch_size = batch_size
        self.files = os.listdir(data_dir)
        self.indexes = np.arange(len(self.files))

    def __len__(self):
        return int(len(self.files)*(4096/self.batch_size)) 

    def __getitem__(self, index):
        with open(self.data_dir+str(index)+'.pkl','rb') as f:
            batch_data=pickle.load(f)

        with open(self.labels_dir+str(index)+'.pkl','rb') as f:
            batch_labels=pickle.load(f)
        
        return np.array(batch_data), np.array(batch_labels)

train_generator = MyGenerator(data_dir='/home/private/Hepd/Dataset_3/batched_dataset/training/bs_4096/data/', labels_dir='/home/private/Hepd/Dataset_3/batched_dataset/training/bs_4096/labels/', batch_size=4096)
validation_generator = MyGenerator(data_dir='/home/private/Hepd/Dataset_3/batched_dataset/validation/bs_4096/data/', labels_dir='/home/private/Hepd/Dataset_3/batched_dataset/validation/bs_4096/labels/', batch_size=4096)

model='fcnn_CNN'
parameters=['271k_par']

for n_par in parameters:
    with open('/home/private/Hepd/'+model+'_model_pre-trained/'+n_par+'/fcnn_CNN.json', 'r') as f:
        modello_json = f.read()    
    fcnn= model_from_json(modello_json)
    optimizer = Adam(learning_rate=0.001)
    fcnn.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    class MyCallback(Callback):
        def __init__(self, threshold, patience):
            super(MyCallback, self).__init__()
            self.threshold = threshold
            self.patience = patience
            self.streak = 0
    
        def on_epoch_end(self, epoch, logs=None):
            val_acc = logs.get('val_accuracy')
            if val_acc is not None and val_acc > self.threshold:
                self.streak += 1
                if self.streak >= self.patience:
                    self.model.stop_training = True
            else:
                self.streak = 0
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.95, patience=5, min_lr=0.0000001)
    my_callback = MyCallback(threshold=0.99999999, patience=5)
    
    
    history = fcnn.fit(train_generator, epochs=1000, shuffle=True,validation_data=validation_generator,validation_steps=len(validation_generator),callbacks=[reduce_lr,my_callback])
    #####
    fcnn.save('/home/private/Hepd/'+model+'_model_pre-trained/'+n_par+'/trained_fcnn.h5')
    last_train_acc = history.history['accuracy'][-1]
    last_train_loss = history.history['loss'][-1]


    with open('/home/private/Hepd/'+model+'_model_pre-trained/'+n_par+'/train_history.txt', 'w') as f:
        f.write(f'Last training accuracy: {last_train_acc}\n')
        f.write(f'Last training loss: {last_train_loss}\n')

    last_val_acc = history.history['val_accuracy'][-1]
    last_val_loss = history.history['val_loss'][-1]

    with open('/home/private/Hepd/'+model+'_model_pre-trained/'+n_par+'/validation_history.txt', 'w') as f:
        f.write(f'Last validation accuracy: {last_val_acc}\n')
        f.write(f'Last validation loss: {last_val_loss}\n')

    with open('/home/private/Hepd/'+model+'_model_pre-trained/'+n_par+'/losses.pkl', 'wb') as f:
        pickle.dump(history.history, f)