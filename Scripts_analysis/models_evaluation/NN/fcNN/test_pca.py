import os

import pickle

import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from matplotlib.ticker import ScalarFormatter


import tensorflow as tf
tf.config.list_physical_devices('GPU')

from keras.utils import Sequence
from keras.models import load_model

#### CREAIAMO LE CLASSI MyGenerator PER FORNIRE IL DATASET AL MODELLO GIà DIVISO IN BATCHES #####

### MyGenerator ASSOCIA AD OGNI EVENTO ANCHE LA SUA LABEL IN MODO DA VALUTARE LE PRESTAZIONI DEL MODELLO (model.evaluate) ###

class MyGenerator_wLables(Sequence):
    
    def __init__(self,data_dir,labels_dir,pca=False,input_dimension=0):
        self.data_dir=data_dir
        self.labels_dir=labels_dir
        self.labels_files=os.listdir(self.labels_dir)
        self.data_files=os.listdir(self.data_dir)

        ##### definiamo una variabile intera per la dimensione dell'input. Ci serve nel caso in cui volessimo escludere alcune event-features, come potremmo fare nel caso di PCA
        # input_dim : int (se è 40 teniamo tutte le feature 31ADC+9LYSO) 
        
        ##### ATTENZIONE !!!!!
        ##### ATTENZIONE !!!!!
        ##### ATTENZIONE !!!!!
        self.pca=pca
        if self.pca:
            self.input_dim=input_dimension
        else:
            self.input_dim=40
        ##### ATTENZIONE !!!!!
        ##### ATTENZIONE !!!!!
        ##### ATTENZIONE !!!!!
        
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

        ### teniamo solo le prime input_dim features
        if self.pca:
            batch_pca_data=[]
            for i in range(len(batch_data)):
                batch_pca_data.append(batch_data[i][:self.input_dim])

            return np.array(batch_pca_data),np.array(batch_labels)
        else:
            return np.array(batch_data),np.array(batch_labels)


### MyGenerator_jData RECUPERA SOLO GLI EVENTI DAL TEST-SET IN MODO CHE IL MODELLO POSSA PREDIRNE L'OUTPUT (model.predict) ###

class MyGnerator_jData(Sequence):

    def __init__(self,data_dir,pca=False,input_dimension=0):
        self.data_dir=data_dir
        self.data_files=os.listdir(self.data_dir)

        ##### definiamo una variabile intera per la dimensione dell'input. Ci serve nel caso in cui volessimo escludere alcune event-features, come potremmo fare nel caso di PCA
        # input_dim : int (se è 40 teniamo tutte le feature 31ADC+9LYSO) 
        
        ##### ATTENZIONE !!!!!
        ##### ATTENZIONE !!!!!
        ##### ATTENZIONE !!!!!
        self.pca=pca
        if self.pca:
            self.input_dim=input_dimension
        else:
            self.input_dim=40
        ##### ATTENZIONE !!!!!
        ##### ATTENZIONE !!!!!
        ##### ATTENZIONE !!!!!
        

        # Assicuriamoci di eliminiare tutti i possibili file nascosti nella cartella dei dati e delle labels
        for i,data in enumerate(self.data_files):
            if data[0]=='.':
                self.data_files.pop(i)

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self,index):
        
        with open(self.data_dir+'/'+str(index)+'.pkl','rb') as f:
            batch_data=pickle.load(f)

        ### teniamo solo le prime input_dim features
        if self.pca:
            batch_pca_data=[]
            for i in range(len(batch_data)):
                batch_pca_data.append(batch_data[i][:self.input_dim])

            return np.array(batch_pca_data)
        else:
            return np.array(batch_data)


##### ATTENZIONE !!!!!
pca=True
##### ATTENZIONE !!!!!

        
###### ISTANZIAMO L'OGGETTO PER FRECUPERARE IL DATASET #####

###### ATTENZIONE AL DATASET GIUSTO SE SERVE O MENO IL PCA !!!!!!!!!


### QUESTO VA DATO A model.evaluate CHE VALUTA LE PRESTAZIONI SUL TEST-SET ###

test_generator_wLabels=MyGenerator_wLables('/home/private/Hepd/Dataset_4/test/test_pca_data','/home/private/Hepd/Dataset_4/test/test_labels',pca=True,input_dimension=4)

### QUESTO VA DATO A model.evaluate CHE VALUTA LE PRESTAZIONI SUL TRAIN-SET ###
train_generator_wLabels=MyGenerator_wLables('/home/private/Hepd/Dataset_4/train/train_pca_data','/home/private/Hepd/Dataset_4/train/train_labels',pca=True,input_dimension=4)

### QUESTO VA DATO A model.evaluate CHE VALUTA LE PRESTAZIONI SUL VALIDATION-SET ###
validation_generator_wLabels=MyGenerator_wLables('/home/private/Hepd/Dataset_4/validation/validation_pca_data','/home/private/Hepd/Dataset_4/validation/validation_labels',pca=True,input_dimension=4)

### QUESTO VA DATA A model.predict CHE PREDICE UN RISULTATO PER OGNI EVENTO NEL TEST-SET ###

test_generator_jData=MyGnerator_jData('/home/private/Hepd/Dataset_4/test/test_pca_data',pca=True,input_dimension=4)

### RECUPERIAMO TUTTE LE LABELS ###
num_lables=len(os.listdir('/home/private/Hepd/Dataset_4/test/test_labels'))
labels=[]
for i in range(num_lables):
    with open('/home/private/Hepd/Dataset_4/test/test_labels/'+str(i)+'.pkl','rb') as f:
        labels.extend(pickle.load(f))


##### RECUPERIAMO TUTTI I MODELLI DA TESTARE #####

#models=os.listdir('fcNN/bottle')+os.listdir('fcNN/funnel')+os.listdir('fcNN/constant')
models=os.listdir('/home/private/Hepd/Dataset_4/fcNN/PCA/bottle')
#models=os.listdir('/home/private/Hepd/Dataset_4/fcNN/PCA/funnel')
#models=os.listdir('/home/private/Hepd/Dataset_4/fcNN/PCA/constant')

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

    
    models_dir.append('/home/private/Hepd/Dataset_4/fcNN/PCA/'+architecture+'/'+model)

for i,dir in enumerate(models_dir):
    ### Carichiamo il modello da testare ###
    model_path=dir+'/'+models[i]+'_trained.h5'
    model=load_model(model_path)

    ### Definiamo il path della cartella in cui salvare i risultati più il riferimento al modello che tutti i file dovranno avere ###
    path=dir+'/'+models[i]
    
    ### Indichiamo a che modello siamo rispetto al totale di modelli ###
    print('Testing the model: '+models[i])
    print('We are testing the '+str(i+1)+'th model with a total of :'+str(len(models_dir)))
    print('\n')
    ### Salviamo su un file .txt accuracy e loss sul test set ###
    metrics=model.evaluate(test_generator_wLabels,verbose=0)
    with open(path+'_test_loss_accuracy.txt','w') as f:
        f.write('Prestazioni sul test set del modello '+str(models[i])+'.\n')
        f.write('Loss :'+str(metrics[0])+'\n')
        f.write('Accuracy :'+str(metrics[1])+'\n') 

    ### Salviamo su un file .txt accuracy e loss sul train set ###
    metrics=model.evaluate(train_generator_wLabels,verbose=0)
    with open(path+'_train_loss_accuracy.txt','w') as f:
        f.write('Prestazioni sul train set del modello '+str(models[i])+'.\n')
        f.write('Loss :'+str(metrics[0])+'\n')
        f.write('Accuracy :'+str(metrics[1])+'\n') 

    ### Salviamo su un file .txt accuracy e loss sul validation set ###
    metrics=model.evaluate(validation_generator_wLabels,verbose=0)
    with open(path+'_validation_loss_accuracy.txt','w') as f:
        f.write('Prestazioni sul validation set del modello '+str(models[i])+'.\n')
        f.write('Loss :'+str(metrics[0])+'\n')
        f.write('Accuracy :'+str(metrics[1])+'\n') 

    ### Valutiamo le predizioni del modello sul test-set ###
    predictions=model.predict(test_generator_jData,verbose=0)
    
    ### Utilizziamo le labels per dividere tra predizioni di Elettroni e di Protoni ###
    predictions=predictions.flatten().tolist()
    electrons_predictions=[]
    protons_predictions=[]
    ### Se la label è 0 si tratta della prediction per un protone, se è 1 si tratta della prediction di un elettrone ###
    for j,pred in enumerate(predictions):
        if labels[j]==0:
            protons_predictions.append(pred)
        elif labels[j]==1:
            electrons_predictions.append(pred)
        else:
            print('LABEL NOT VALID')
    ### Salviamo le predizioni totali e specifiche per elettroni e protoni ###
    with open(path+'_predictions.pkl','wb') as f:
        pickle.dump(predictions,f)

    with open(path+'_electrons_predictions.pkl','wb') as f:
        pickle.dump(electrons_predictions,f)

    with open(path+'_protons_prediction.pkl','wb') as f:
        pickle.dump(protons_predictions,f)

    ### Creiamo le ROC ###
    fpr, tpr,thresholds = roc_curve(labels, predictions)
    roc_auc = auc(fpr, tpr)
    
    ### Salviamo i false_positive_rate, i true_positive_rate e i trhesholds ###
    with open(path+'_test_fpr.pkl','wb') as f:
        pickle.dump(fpr,f)

    with open(path+'_test_tpr.pkl','wb') as f:
        pickle.dump(tpr,f)

    with open(path+'_test_thresholds.pkl','wb') as f:
        pickle.dump(thresholds,f)

    with open(path+'_test_ROC_auc.pkl','wb') as f:
        pickle.dump(roc_auc,f)

print('Done')