import os

import pickle

import numpy as np

import tensorflow as tf
tf.config.list_physical_devices('GPU')

from keras.utils import Sequence
from keras.models import load_model

### CREIAMO LA CLASSE MyGenerator PER FORNIRE I DATI DEL TRAIN E DEL VALIDATION SET AL MODELLO  ###
### VOGLIAMO VALUTARE CON IL BEST MODEL LE PERFORMANCE SU TRAIN E VALIDATION SET ###

class MyGenerator(Sequence):
    
    def __init__(self,data_dir,labels_dir):
        self.data_dir=data_dir
        self.labels_dir=labels_dir
        self.labels_files=os.listdir(self.labels_dir)
        self.data_files=os.listdir(self.data_dir)
        
        # Assicuriamoci di eliminiare tutti i possibili file nascosti nella cartella dei dati e delle labels
        for i,data in enumerate(self.data_files):
            if data[0]=='.':
                self.data_files.pop(i)
                
        for i,labels in enumerate(self.labels_files):
            if data[0]=='.':
                self.labels_files.pop(i)
        
        with open(self.labels_dir+'/'+self.labels_files[0],'rb') as f:
            self.batch_size=len(pickle.load(f))

        self.indexes=np.arange(len(self.labels_files))
    
    def __len__(self):
        return len(self.data_files)

    def __getitem__(self,index):
        
        with open(self.data_dir+'/'+str(index)+'.pkl','rb') as f:
            batch_data=pickle.load(f)

        with open(self.labels_dir+'/'+str(index)+'.pkl','rb') as f:
            batch_labels=pickle.load(f)

        return np.array(batch_data),np.array(batch_labels)
### ISTANZIAMO GLI OGGETTI DA PASSRE A model.evaluate ###
train_generator=MyGenerator('train/train_norm_data','train/train_labels')

validation_generator=MyGenerator('validation/validation_norm_data','validation/validation_labels')

##### RECUPERIAMO TUTTI I MODELLI DA VALUTARE #####

models=os.listdir('fcNN/bottle')+os.listdir('fcNN/funnel')+os.listdir('fcNN/constant')
models_dir=[]
for model in models:
    # individuiamo la forma dell'architettura: la prima lettera di model ci permette di individuare di che tipo di arhcitettura si tratta
    if model[0]=='b':
        architecture='bottle'
    elif model[0]=='f':
        architecture='funnel'
    elif model[0]=='c':
        architecture='constant'
    elif model[0]=='g':
        architecture='general'
    models_dir.append('fcNN/'+architecture+'/'+model)

for i,dir in enumerate(models_dir):
    ### Carichiamo il modello da testare ###
    model_path=dir+'/'+models[i]+'_trained.h5'
    model=load_model(model_path)

    ### Definiamo il path della cartella in cui salvare i risultati più il riferimento al modello che tutti i file dovranno avere ###
    path=dir+'/'+models[i]

    ### Valutiamo i modelli 
    train_metrics=model.evaluate(train_generator)
    validation_metrics=model.evaluate(validation_generator)

    ### Salviamo i risultati
    with open(path+'_train_loss_value.pkl','wb') as f:
        pickle.dump(train_metrics[0],f)

    with open(path+'_train_accuracy_value.pkl','wb') as f:
        pickle.dump(train_metrics[1],f)

    with open(path+'_validation_loss_value.pkl','wb') as f:
        pickle.dump(validation_metrics[0],f)

    with open(path+'_validation_accuracy_value.pkl','wb') as f:
        pickle.dump(validation_metrics[1],f)