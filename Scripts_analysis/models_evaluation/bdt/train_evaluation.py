#### per valutare l'overfitting calcoliamo l'area delle ROC di tutte le bdt sul training set
#### andremo a confrontarle con le aree sotto il test set

import lightgbm as lgb
import os
import pickle
import joblib
import numpy as np
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings("ignore")

#### testiamo e salviamo i risultati per tutte le bdt

#### carichiamo il train set, nota bene per le BDT non utilizziamo il dataset normalizzato ####
train_data_dir='/home/private/Hepd/Dataset_4/train/train_data/'
train_labels_dir='/home/private/Hepd/Dataset_4/train/train_labels/'
train_data=[]
train_labels=[]
train_batches=os.listdir(train_data_dir)

for batch in train_batches:
    with open(train_data_dir+batch,'rb') as f:
        train_data.extend(pickle.load(f))
    with open(train_labels_dir+batch,'rb') as f:
        train_labels.extend(pickle.load(f))

#### creiamo una lista con tutti modelli di bdt da testare
path_dir = '/home/private/Hepd/Dataset_4/bdt'
model_list = os.listdir(path_dir)
#### testaiamo su tutti i modelli e salviamo le prediction
#### a differenza di quello che abbiamo fatto sul test set qui guardiamo solo le probabilità di appartenere alla
#### alla classe signal, abbiamo visto che la sigmoid del raw score è un pessimo selettore
for j,model_name in enumerate(model_list):
    electrons_probabilities = []
    protons_probabilities = []
    print(f'stiamo testando il modello {model_name}, è il numero {j+1} su {len(model_list)}')
    ### carichiamo il modello salvato e trainato
    model = joblib.load(path_dir+'/'+model_name+'/'+model_name+'_trained.pkl')
    ### otteniamo i valori delle due probabilità associate ad ogni evento
    p = model.predict_proba(train_data)
    ### in base al valore della label ci salviamo l'output della bdt per gli elettroni o per i protoni
    for i,label in enumerate(train_labels):
        if label == 1:
            electrons_probabilities.append(p[i])
        elif label == 0:
            protons_probabilities.append(p[i])
        else:
            print(f'label problem at event {i}')
    probabilities = np.array(p)
    protons_probabilities = np.array(protons_probabilities)
    electrons_probabilities = np.array(electrons_probabilities)
    #### ci salviamo queste queste probabilità
    with open(path_dir+'/'+model_name+'/'+model_name+'_train_probabilities.pkl','wb') as f:
        pickle.dump(probabilities,f)
    with open(path_dir+'/'+model_name+'/'+model_name+'_train_electrons_probabilities.pkl','wb') as f:
        pickle.dump(electrons_probabilities,f)
    with open(path_dir+'/'+model_name+'/'+model_name+'_train_protons_probabilities.pkl','wb') as f:
        pickle.dump(protons_probabilities,f)

    #### calcoliamo la roc curve e la sua area
    probabilities_1 = probabilities[:,1]
    fpr_probabilities_1, tpr_probabilities_1, thresholds_probabilities_1 = roc_curve(train_labels, probabilities_1)
    auc_probabilities_1 = auc(fpr_probabilities_1,tpr_probabilities_1)
    ### salviamo tutto
    with open(path_dir+'/'+model_name+'/'+model_name+'_train_fpr_probabilities_1.pkl','wb') as f:
        pickle.dump(fpr_probabilities_1,f)
    with open(path_dir+'/'+model_name+'/'+model_name+'_train_tpr_probabilities_1.pkl','wb') as f:
        pickle.dump(tpr_probabilities_1,f)
    with open(path_dir+'/'+model_name+'/'+model_name+'_train_thresholds_probabilities_1.pkl','wb') as f:
        pickle.dump(thresholds_probabilities_1,f)
    with open(path_dir+'/'+model_name+'/'+model_name+'_train_ROC_area_probabilities_1.txt','w') as f:
        f.write(f'Area sotto la ROC calcolata usando come selettore la probabilità della classe 1 (protone) assegnata dalla BDT sul train set: {auc_probabilities_1}')

    print('stiamo calcolando lo score') 
    #### ci calcoliamo e salviamo lo score della bdt, ossia la mean accuracy
    s = model.score(train_data,train_labels)
    with open(path_dir+'/'+model_name+'/'+model_name+'_train_accuracy.txt','w') as f:
        f.write(f'Model mean train accuracy: {s}')