#### creaimo e salviamo i plot delle predictions e delle roc per tutte le bdt
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from sklearn.metrics import roc_curve, auc

### apriamo le labels del test set
test_labels_dir='/home/private/Hepd/Dataset_4/test/test_labels/'
test_labels=[]
test_batches=os.listdir(test_labels_dir)

### otteniamo la lista dei modelli
path = '/home/private/Hepd/Dataset_4/bdt'
model_list = os.listdir(path)


### creiamo tutti i plot per tutti i modelli
for i,model_name in enumerate(model_list):
    print(f'Stiamo plottando il modello {model_name} che è il numero {i+1} su un totale {len(model_list)}')
    ### creiamo il plot della distribuzione della sigmoid 
    with open(path+'/'+model_name+'/'+model_name+'_electrons_predictions.pkl','rb') as f:
        electrons_predictions_sigmoid = pickle.load(f)
    with open(path+'/'+model_name+'/'+model_name+'_protons_predictions.pkl','rb') as f:
        protons_predictions_sigmoid = pickle.load(f)
    plt.figure()
    plt.hist(electrons_predictions_sigmoid, color = 'b', bins = 1000, label = 'Electrons')
    plt.hist(protons_predictions_sigmoid, color = 'r', bins = 1000, label = 'Protons', alpha = 0.5)
    plt.xlabel('Predictions sigmoid')
    plt.ylabel('Frequency')
    plt.title(f'Predictions sigmoid distribution\n model: {model_name}')
    plt.legend(loc = 'best')
    plt.savefig(path+'/'+model_name+'/'+model_name+'_predictions_sigmoid_distribution_plot.png')
    plt.close()

    ### creiamo lo stesso plot in scala logaritmica
    plt.figure()
    plt.hist(electrons_predictions_sigmoid, color = 'b', bins = 1000, label = 'Electrons')
    plt.hist(protons_predictions_sigmoid, color = 'r', bins = 1000, label = 'Protons', alpha = 0.5)
    plt.xlabel('Predictions sigmoid')
    plt.ylabel('Frequency')
    plt.title(f'Predictions sigmoid distribution log scale\n model: {model_name}')
    plt.yscale('log')
    plt.legend(loc = 'best')
    plt.savefig(path+'/'+model_name+'/'+model_name+'_predictions_sigmoid_distribution_log_plot.png')
    plt.close()
    
    ### creiamo il plot della distribuzione della probabilità di classe 0 e 1
    with open(path+'/'+model_name+'/'+model_name+'_electrons_probabilities.pkl','rb') as f:
        electrons_probabilities = pickle.load(f)
    with open(path+'/'+model_name+'/'+model_name+'_protons_probabilities.pkl','rb') as f:
        protons_probabilities = pickle.load(f)
    ### electrons_probabilities e protons_probabilities sono array bidimensionali
    ### i primi elementi sono la probabilità di quell'evento di essere elettrone o protone che viene assegnata dalla BDT
    ### electrons_probabilities_0 : array[float] probabilità assegnata agli elettroni di essere protoni dalla BDT
    electrons_probabilities_0 = electrons_probabilities[:,0]
    ### electrons_probabilities_1 : array[float] probabilità assegnata agli elettroni di essere elettroni dalla BDT
    electrons_probabilities_1 = electrons_probabilities[:,1]
    ### protons_probabilities_0 : array[float] probabilità assegnata ai protoni di essere protoni dalla BDT
    protons_probabilities_0 = protons_probabilities[:,0]
    ### protons_probabilities_1 : array[float] probabilità assegnata ai protoni di essere elettroni dalla BDT
    protons_probabilities_1 = protons_probabilities[:,1]

    ### plottiamo le distribuzioni di probabilità assegnate alla classe signal (electrons)
    plt.figure()
    plt.hist(electrons_probabilities_1, color = 'b', bins = 1000, label = 'Electrons')
    plt.hist(protons_probabilities_1, color = 'r', bins = 1000, label = 'Protons', alpha = 0.5)
    plt.title(f'Signal probability distribution\n model: {model_name}')
    plt.xlabel('Signal probability')
    plt.ylabel('Frequency')
    plt.legend(loc = 'best')
    plt.savefig(path+'/'+model_name+'/'+model_name+'_signal_probability_distribution_plot.png')
    plt.close()

    ### creiamo lo stesso plot in scala logaritmica
    plt.figure()
    plt.hist(electrons_probabilities_1, color = 'b', bins = 1000, label = 'Electrons')
    plt.hist(protons_probabilities_1, color = 'r', bins = 1000, label = 'Protons', alpha = 0.5)
    plt.title(f'Signal probability distribution log scale\n model: {model_name}')
    plt.xlabel('Signal probability')
    plt.ylabel('Frequency')
    plt.yscale('log')
    plt.legend(loc = 'best')
    plt.savefig(path+'/'+model_name+'/'+model_name+'_signal_probability_distribution_log_plot.png')
    plt.close()

    ### apriamo le tpr e le fpr ottenute con selettore predictions_sigmoid
    with open(path+'/'+model_name+'/'+model_name+'_fpr_predictions_sigmoid.pkl','rb') as f:
        fpr_predictions_sigmoid = pickle.load(f)
    with open(path+'/'+model_name+'/'+model_name+'_tpr_predictions_sigmoid.pkl','rb') as f:
        tpr_predictions_sigmoid = pickle.load(f)
    #### invece che estrarre l'area sotto la roc dal file txt la ricalcoliamo
    auc_predictions_sigmoid = auc(fpr_predictions_sigmoid,tpr_predictions_sigmoid)
    ### usiamo l'area della loss da mettere nella label della plot della ROC
    plt.figure()
    plt.plot(tpr_predictions_sigmoid,1-fpr_predictions_sigmoid, label = 'ROC area  = %0.6f' % auc_predictions_sigmoid, color = 'green')
    plt.title(f'ROC with predictions sigmoid over test set\n model: {model_name}')
    plt.legend(loc = 'best')
    plt.ylabel('Background rejection')
    plt.xlabel('Signal efficiency')
    plt.savefig(path+'/'+model_name+'/'+model_name+'_roc_predictions_sigmoid.png')
    plt.close()
    
    ### apriamo le tpr e le fpr ottenute con selettore probabilities_1
    with open(path+'/'+model_name+'/'+model_name+'_fpr_probabilities_1.pkl','rb') as f:
        fpr_probabilities_1 = pickle.load(f)
    with open(path+'/'+model_name+'/'+model_name+'_tpr_probabilities_1.pkl','rb') as f:
        tpr_probabilities_1 = pickle.load(f)
    #### invece che estrarre l'area sotto la roc dal file txt la ricalcoliamo
    auc_probabilities_1 = auc(fpr_probabilities_1,tpr_probabilities_1)
    ### usiamo l'area della loss da mettere nella label della plot della ROC
    plt.figure()
    plt.plot(tpr_probabilities_1,1-fpr_probabilities_1, label = 'ROC area  = %0.6f' % auc_probabilities_1, color = 'green')
    plt.title(f'ROC with signal probabilities over test set\n model: {model_name}')
    plt.ylabel('Background rejection')
    plt.xlabel('Signal efficiency')
    plt.legend(loc = 'best')
    plt.savefig(path+'/'+model_name+'/'+model_name+'_roc_probabilities_1.png')
    plt.close()