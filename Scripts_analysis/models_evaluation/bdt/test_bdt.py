import lightgbm as lgb
import os
import pickle
import joblib
import numpy as np
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings("ignore")

#### testiamo e salviamo i risultati per tutte le bdt

#### carichiamo il test set, nota bene per le BDT on utilizziamo il dataset normalizzato ####
test_data_dir='/home/private/Hepd/Dataset_4/test/test_data/'
test_labels_dir='/home/private/Hepd/Dataset_4/test/test_labels/'
test_data=[]
test_labels=[]
test_batches=os.listdir(test_data_dir)
for batch in test_batches:
    with open(test_data_dir+batch,'rb') as f:
        test_data.extend(pickle.load(f))
    with open(test_labels_dir+batch,'rb') as f:
        test_labels.extend(pickle.load(f))
#### nel caso della bdt potremmo concatenare al test set anche il validation set ?

#### definiamo la funzione sigmoide, ci servirà per normalizzare il raw_score delle bdt
def sigmoid(x):
    return 1/(1+np.exp(-x))

#### creiamo una lista con tutti modelli di bdt da testare
path_dir = '/home/private/Hepd/Dataset_4/bdt'
model_list = os.listdir(path_dir)
#### testaiamo su tutti i modelli e salviamo le prediction
for j,model_name in enumerate(model_list):
    electrons_predictions = []
    protons_predictions = []
    electrons_probabilities = []
    protons_probabilities = []
    print(f'stiamo testando il modello {model_name}, è il numero {j+1} su {len(model_list)}')
    ### carichiamo il modello salvato e trainato
    model = joblib.load(path_dir+'/'+model_name+'/'+model_name+'_trained.pkl')
    ### otteniamo il raw_value associato ad ogni evento e ne clcoliamo la sigmoide
    k = model.predict(test_data)
    k = sigmoid(k)
    ### otteniamo i valori delle due probabilità associate ad ogni evento
    p = model.predict_proba(test_data)
    ### in base al valore della label ci salviamo l'output della bdt per gli elettroni o per i protoni
    for i,label in enumerate(test_labels):
        if label == 1:
            electrons_predictions.append(k[i])
            electrons_probabilities.append(p[i])
        elif label == 0:
            protons_predictions.append(k[i])
            protons_probabilities.append(p[i])
        else:
            print(f'label problem at event {i}')
    probabilities = np.array(p)
    predictions = np.array(k)
    protons_predictions = np.array(protons_predictions)
    electrons_predictions = np.array(electrons_predictions)
    protons_probabilities = np.array(protons_probabilities)
    electrons_probabilities = np.array(electrons_probabilities)
    #### ci salviamo queste predictions e queste probabilità
    with open(path_dir+'/'+model_name+'/'+model_name+'_predictions.pkl','wb') as f:
        pickle.dump(predictions,f)
    with open(path_dir+'/'+model_name+'/'+model_name+'_probabilities.pkl','wb') as f:
        pickle.dump(probabilities,f)
    with open(path_dir+'/'+model_name+'/'+model_name+'_electrons_predictions.pkl','wb') as f:
        pickle.dump(electrons_predictions,f)
    with open(path_dir+'/'+model_name+'/'+model_name+'_electrons_probabilities.pkl','wb') as f:
        pickle.dump(electrons_probabilities,f)
    with open(path_dir+'/'+model_name+'/'+model_name+'_protons_predictions.pkl','wb') as f:
        pickle.dump(protons_predictions,f)
    with open(path_dir+'/'+model_name+'/'+model_name+'_protons_probabilities.pkl','wb') as f:
        pickle.dump(protons_probabilities,f)
    ### calcoliamo fpr, tpr, thresholds e area sotto la ROC curve e li salviamo
    ### creiamo due ROC: 
    # la prima utilizzando la sigmoid della prediction
    fpr_prediction_sigmoid, tpr_prediction_sigmoid, thresholds_prediction_sigmoid = roc_curve(test_labels, predictions)
    auc_predictions = auc(fpr_prediction_sigmoid,tpr_prediction_sigmoid)
    ### salviamo tutto
    with open(path_dir+'/'+model_name+'/'+model_name+'_fpr_predictions_sigmoid.pkl','wb') as f:
        pickle.dump(fpr_prediction_sigmoid,f)
    with open(path_dir+'/'+model_name+'/'+model_name+'_tpr_predictions_sigmoid.pkl','wb') as f:
        pickle.dump(tpr_prediction_sigmoid,f)
    with open(path_dir+'/'+model_name+'/'+model_name+'_thresholds_prediction_sigmoid.pkl','wb') as f:
        pickle.dump(thresholds_prediction_sigmoid,f)
    with open(path_dir+'/'+model_name+'/'+model_name+'_ROC_area_predictions_sigmoid.txt','w') as f:
        f.write(f'Area sotto la ROC calcolata usando come selettore la sigmoide del Raw Score della BDT: {auc_predictions}')

    # la seconda utilizzando la probabilità della classe 1, sarà quasi 1 per eventi di segnale (elettroni) che hanno label 1 e quasi 0 per eventi di background (protoni)
    # che hanno label 0, quindi non c'è bisogno di fare nulla e possiamo calcolare fpr e tpr direttamente
    probabilities_1 = probabilities[:,1]
    fpr_probabilities_1, tpr_probabilities_1, thresholds_probabilities_1 = roc_curve(test_labels, probabilities_1)
    auc_probabilities_1 = auc(fpr_probabilities_1,tpr_probabilities_1)
    ### salviamo tutto
    with open(path_dir+'/'+model_name+'/'+model_name+'_fpr_probabilities_1.pkl','wb') as f:
        pickle.dump(fpr_probabilities_1,f)
    with open(path_dir+'/'+model_name+'/'+model_name+'_tpr_probabilities_1.pkl','wb') as f:
        pickle.dump(tpr_probabilities_1,f)
    with open(path_dir+'/'+model_name+'/'+model_name+'_thresholds_probabilities_1.pkl','wb') as f:
        pickle.dump(thresholds_probabilities_1,f)
    with open(path_dir+'/'+model_name+'/'+model_name+'_ROC_area_probabilities_1.txt','w') as f:
        f.write(f'Area sotto la ROC calcolata usando come selettore la probabilità della classe 1 (protone) assegnata dalla BDT: {auc_probabilities_1}')

    print('stiamo calcolando lo score') 
    #### ci calcoliamo e salviamo lo score della bdt, ossia la mean accuracy
    s = model.score(test_data,test_labels)
    with open(path_dir+'/'+model_name+'/'+model_name+'_accuracy.txt','w') as f:
        f.write(f'Model mean accuracy: {s}')