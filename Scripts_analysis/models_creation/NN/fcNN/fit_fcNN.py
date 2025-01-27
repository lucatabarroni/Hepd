import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
tf.config.list_physical_devices('GPU')

import keras

from keras.utils import Sequence
from keras.models import model_from_json
from tensorflow.keras.optimizers import Adam, RMSprop
from keras.callbacks import ReduceLROnPlateau, Callback, EarlyStopping


# DEFINIAMO LA CLASSE MyGenerator CON CUI FORNIAMO I DATI DIVISI IN BATCHES AL METODO FIT

class MyGenerator(Sequence):
    
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

        with open(self.labels_dir+'/'+str(index)+'.pkl','rb') as f:
            batch_labels=pickle.load(f)
        
        with open(self.data_dir+'/'+str(index)+'.pkl','rb') as f:
            batch_data=pickle.load(f)

        ### teniamo solo le prime input_dim features
        if self.pca:
            batch_pca_data=[]
            for i in range(len(batch_data)):
                batch_pca_data.append(batch_data[i][:self.input_dim])

            return np.array(batch_pca_data),np.array(batch_labels)
        else:
            return np.array(batch_data),np.array(batch_labels)

#ISTANZIAMO GLI OGGETTI SIA PER IL TRAINING SET CHE PER IL VALIDATION SET

##### ATTENZIONE !!!!!
pca=True
##### ATTENZIONE !!!!!


#train_generator=MyGenerator('/home/private/Hepd/Dataset_4/train/train_norm_data','/home/private/Hepd/Dataset_4/train/train_labels',pca=False)
#validation_generator=MyGenerator('/home/private/Hepd/Dataset_4/validation/validation_norm_data','/home/private/Hepd/Dataset_4/validation/validation_labels',pca=False)

train_generator=MyGenerator('/home/private/Hepd/Dataset_4/train/train_pca_data','/home/private/Hepd/Dataset_4/train/train_labels',pca=True,input_dimension=4)
validation_generator=MyGenerator('/home/private/Hepd/Dataset_4/validation/validation_pca_data','/home/private/Hepd/Dataset_4/validation/validation_labels',pca=True,input_dimension=4)

# DEFINIAMO I CALLABACK CHE POTRANNO ESSERE UTILIZZATI DAL METODO FIT


# Il metodo BestValidationLossCallback a fine train carica i pesi che durante l'addestramento hanno minimizzato la validation loss
class BestValidationLossCallback(Callback):
    def __init__(self):
        super(BestValidationLossCallback, self).__init__()
        self.best_weights = None
        self.best_val_loss = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get('val_loss')
        if val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_weights = self.model.get_weights()

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)
            print(f"Caricati i pesi del modello con la perdita di validazione migliore: {self.best_val_loss}")



# Il metodo LossAccuracyHistory registra le train e validation loss e accuracy durante le epoche di addestramento
class LossAccuracyHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accuracies = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accuracies.append(logs.get('accuracy'))



# Il metodo ReduceLROnPlateau riduce il learning rate in base alla validation loss:
# se ogni 5 epoche non diminuisce la val_loss il learning rate viene diminuito del 5% fino ad un minimo di 10e-6
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',  # monitor: string indica la metrica da monitorare durante l'addestramento
    factor=0.95,         # factor: float coefficente con cui diminuiamo il learning rate lr-> factor*lr 
    patience=5,          # patience: int numero di epoche da attendere prima di dimunire il learning rate
    min_lr=0.0000001     # float valore minimo del learning_rate
)


# Il callback EarlyStopping serve per interrompere l'addestramento quando una metrica monitorata non migliora
# se ogni 15 epoche la train_loss non diminuisce si interrompe l'addestramento
early_stopping = EarlyStopping(
    monitor='loss',       # monitor: string la metrica da monitorare
    min_delta=0.0001,     # min_delta: float la variazione minima per considerare un miglioramento
    patience=15,          # patience: int numero di epoche con nessun miglioramento dopo le quali interrompere l'allenamento
    verbose=1,            # verbose: int livello di verbosità
    mode='min'            # mode: string modalità di monitoraggio ('min' per minimizzare la metrica monitorata)
)

# Creiamo una lista di stringhe che contengono gli indirizzi dei modelli dai addestrare

##### ATTENZIONE SE DEVI ADDRESTARE IO MODELLI PCA DEVI CAMBIARE GLI INDIRIZZI

#models=os.listdir('/home/private/Hepd/Dataset_4/fcNN/PCA/bottle')
#models=os.listdir('/home/private/Hepd/Dataset_4/fcNN/PCA/funnel')
models=os.listdir('/home/private/Hepd/Dataset_4/fcNN/PCA/constant')


# dobbiamo eliminare eventuali files nascosti nella cartella dei modelli

for i,model in enumerate(models):
    if model[0]=='.':
        models.pop(i)

# dobbiamo fare in modo di avere una lista con gli infirzzi delle dirctory che contengono i modelli da addestrare
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
        
##### ATTENZIONE SE DEVI ADDRESTARE IO MODELLI PCA DEVI CAMBIARE GLI INDIRIZZI
    models_dir.append('/home/private/Hepd/Dataset_4/fcNN/PCA/'+architecture+'/'+model)
        


# ITERIAMO SU TUTTI I MODELLI DA ADDESTRARE
for i,dir in enumerate(models_dir):
    print('Train modello: ',dir,' è il numero '+str(i+1)+' su ' +str(len(models_dir)))

    # Carichiamo il modello non trainato salvato come .json
    with open(dir+'/'+models[i]+'.json','r') as f:
        model_json=f.read()

    # Compiliamo il modello non trainato
    model=model_from_json(model_json)
    
    #optimizer = RMSprop(learning_rate=0.001)
    optimizer=Adam(learning_rate=0.01)
    model.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])

    # Istanziamo i callback, quelle di BestValidationLossCallabck e LossAccuracyHistory (quest'ultimo non sono sicuro)
    # devono essere istanziate una volta per ogni modello
    load_best_val_loss=BestValidationLossCallback()
    #history_in_epoch=LossAccuracyHistory()    

    # Addestriamo il modello utilizzando i Callback e salvando la storia durante l'Addestramento
    history=model.fit(train_generator,epochs=200,shuffle=True,verbose=0,validation_data=validation_generator,validation_steps=len(validation_generator),
                      callbacks=[load_best_val_loss,early_stopping])

    # Salviamo il modello nella stessa cartella ma aggiungendo _trained e in formato .h5
    model.save(dir+'/'+models[i]+'_trained.h5')

    # Salviamo l'history dell'addestramento sempre nella stessa cartella
    with open(dir+'/'+models[i]+'_history.pkl','wb')as f:
        pickle.dump(history.history,f)
