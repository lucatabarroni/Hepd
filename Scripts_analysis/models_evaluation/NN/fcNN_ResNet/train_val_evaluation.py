import os

import pickle

import numpy as np

from sklearn.metrics import roc_curve, auc

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

### MyGenerator_jData RECUPERA SOLO GLI EVENTI DAL TEST-SET IN MODO CHE IL MODELLO POSSA PREDIRNE L'OUTPUT (model.predict) ###

class MyGenerator_jData(Sequence):

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
### ISTANZIAMO GLI OGGETTI DA PASSRE A model.evaluate ###
train_generator=MyGenerator('/home/private/Hepd/Dataset_4/train/train_norm_data','/home/private/Hepd/Dataset_4/train/train_labels')

validation_generator=MyGenerator('/home/private/Hepd/Dataset_4/validation/validation_norm_data','/home/private/Hepd/Dataset_4/validation/validation_labels')

#### istanzioam gli oggetti da passare a model.predict
train_generatorjData = MyGenerator_jData('/home/private/Hepd/Dataset_4/train/train_norm_data')

validation_generatorjData = MyGenerator_jData('/home/private/Hepd/Dataset_4/validation/validation_norm_data')

### RECUPERIAMO TUTTE LE LABELS ###
num_lables=len(os.listdir('/home/private/Hepd/Dataset_4/train/train_labels'))
train_labels=[]
for i in range(num_lables):
    with open('/home/private/Hepd/Dataset_4/train/train_labels/'+str(i)+'.pkl','rb') as f:
        train_labels.extend(pickle.load(f))


num_lables=len(os.listdir('/home/private/Hepd/Dataset_4/validation/validation_labels'))
validation_labels=[]
for i in range(num_lables):
    with open('/home/private/Hepd/Dataset_4/validation/validation_labels/'+str(i)+'.pkl','rb') as f:
        validation_labels.extend(pickle.load(f))

##### RECUPERIAMO TUTTI I MODELLI DA VALUTARE #####

models=os.listdir('/home/private/Hepd/Dataset_4/fcNN/normalized/bottle')+os.listdir('/home/private/Hepd/Dataset_4/fcNN/normalized/funnel')+os.listdir('/home/private/Hepd/Dataset_4/fcNN/normalized/constant')
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
    models_dir.append('/home/private/Hepd/Dataset_4/fcNN/normalized/'+architecture+'/'+model)

for i,model_name in enumerate(models_dir):
    print(f'We are evaluating the model {model_name} the number {i} on a total of {len(models_dir)}')
    ### Carichiamo il modello da testare ###
    model_path=model_name+'/'+models[i]+'_trained.h5'
    model=load_model(model_path)

    ### Definiamo il path della cartella in cui salvare i risultati più il riferimento al modello che tutti i file dovranno avere ###
    path=model_name+'/'+models[i]

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

    #### creaimo le ROC su train sets
    train_predictions = model.predict(train_generatorjData)
    
    ### Utilizziamo le labels per dividere tra predizioni di Elettroni e di Protoni ###
    train_predictions=train_predictions.flatten().tolist()
    train_electrons_predictions=[]
    train_protons_predictions=[]
    ### Se la label è 0 si tratta della prediction per un protone, se è 1 si tratta della prediction di un elettrone ###
    for j,pred in enumerate(train_predictions):
        if train_labels[j]==0:
            train_protons_predictions.append(pred)
        elif train_labels[j]==1:
            train_electrons_predictions.append(pred)
        else:
            print('LABEL NOT VALID')
    ### Salviamo le predizioni totali e specifiche per elettroni e protoni ###
    with open(path+'_train_predictions.pkl','wb') as f:
        pickle.dump(train_predictions,f)

    with open(path+'_train_electrons_predictions.pkl','wb') as f:
        pickle.dump(train_electrons_predictions,f)

    with open(path+'_train_protons_prediction.pkl','wb') as f:
        pickle.dump(train_protons_predictions,f)

    ### Creiamo le ROC ###
    fpr, tpr,thresholds = roc_curve(train_labels, train_predictions)
    roc_auc = auc(fpr, tpr)
    
    ### Salviamo i false_positive_rate, i true_positive_rate e i trhesholds ###
    with open(path+'_train_fpr.pkl','wb') as f:
        pickle.dump(fpr,f)

    with open(path+'_train_tpr.pkl','wb') as f:
        pickle.dump(tpr,f)

    with open(path+'_train_thresholds.pkl','wb') as f:
        pickle.dump(thresholds,f)

    with open(path+'_train_ROC_auc.pkl','wb') as f:
        pickle.dump(roc_auc,f)


    #### creaimo le ROC validation sets
    validation_predictions = model.predict(validation_generatorjData)
    
    ### Utilizziamo le labels per dividere tra predizioni di Elettroni e di Protoni ###
    validation_predictions=validation_predictions.flatten().tolist()
    validation_electrons_predictions=[]
    validation_protons_predictions=[]
    ### Se la label è 0 si tratta della prediction per un protone, se è 1 si tratta della prediction di un elettrone ###
    for j,pred in enumerate(validation_predictions):
        if validation_labels[j]==0:
            validation_protons_predictions.append(pred)
        elif validation_labels[j]==1:
            validation_electrons_predictions.append(pred)
        else:
            print('LABEL NOT VALID')
    ### Salviamo le predizioni totali e specifiche per elettroni e protoni ###
    with open(path+'_validation_predictions.pkl','wb') as f:
        pickle.dump(validation_predictions,f)

    with open(path+'_validation_electrons_predictions.pkl','wb') as f:
        pickle.dump(validation_electrons_predictions,f)

    with open(path+'_validation_protons_prediction.pkl','wb') as f:
        pickle.dump(validation_protons_predictions,f)

    ### Creiamo le ROC ###
    fpr, tpr,thresholds = roc_curve(validation_labels, validation_predictions)
    roc_auc = auc(fpr, tpr)
    
    ### Salviamo i false_positive_rate, i true_positive_rate e i trhesholds ###
    with open(path+'_validation_fpr.pkl','wb') as f:
        pickle.dump(fpr,f)

    with open(path+'_validation_tpr.pkl','wb') as f:
        pickle.dump(tpr,f)

    with open(path+'_validation_thresholds.pkl','wb') as f:
        pickle.dump(thresholds,f)

    with open(path+'_validation_ROC_auc.pkl','wb') as f:
        pickle.dump(roc_auc,f)