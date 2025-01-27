import pickle
import numpy as np
import os

#### CERCHIAMO LA MASSIMA ROC AREA ####

#### DEFINIAMO UNA AREA MASSIMA INIZIALE DI 0.5 (LA MINIMA POSSIBILE PER UNA ROC) ####

max_area=0.5
best_model=[]

#### DEFINIAMO ANCHE UN MASSIMO PER OGNI ARCHITETTURA IN MODO DA TROVARE IL MIGLIOR MODELLO PER OGNI ARCHITETTURA ####

max_area_bottle=0.5
best_bottle_model=[]

max_area_funnel=0.5
best_funnel_model=[]

max_area_constant=0.5
best_constant_model=[]

#### OTTENIAMO LA LISTA DI TUTTE LE DIRECTORY DA CUI ESTRARRE LE ROC AREA ####

archs=['bottle','funnel','constant']

for arch in archs:
    ### lista dei modelli per ogni architettura ###
    path='/home/private/Hepd/Dataset_4/fcNN/PCA/'+arch
    models=os.listdir(path)
    for model in models:
        #### otteniamo il path del file della roc ####
        roc_file=path+'/'+model+'/'+model+'_test_ROC_auc.pkl'
        with open(roc_file,'rb') as f:
            area=pickle.load(f)
        #### se l'area è maggiore ad max_area salviamo questo nuovo modello con area massima per la test roc ####
        if area>=max_area:
            max_area=area
            best_model=model
        #### cerchiamo il modello con la roc area massima per ogni architettura ####
        if arch=='bottle':
            if area>=max_area_bottle:
                max_area_bottle=area
                best_bottle_model=model
        elif arch=='funnel':
            if area>=max_area_funnel:
                max_area_funnel=area
                best_funnel_model=model
        else:
            if area>=max_area_constant:
                max_area_constant=area
                best_constant_model=model

with open('best_roc_pca.txt','w') as f:
    f.write('Best model: '+best_model+'\n')
    f.write(f'ROC area {max_area}')

with open('/home/private/Hepd/Dataset_4/analysis/PCA/bottle/bottle_pca_best_roc.txt','w') as f:
    f.write('Best bottle PCA model: '+best_bottle_model+'\n')
    f.write(f'ROC area {max_area_bottle}')

with open('/home/private/Hepd/Dataset_4/analysis/PCA/funnel/funnel_pca_best_roc.txt','w') as f:
    f.write('Best funnel PCA model: '+best_funnel_model+'\n')
    f.write(f'ROC area {max_area_funnel}')

with open('/home/private/Hepd/Dataset_4/analysis/PCA/constant/constant_pca_best_roc.txt','w') as f:
    f.write('Best constant PCA model: '+best_constant_model+'\n')
    f.write(f'ROC area {max_area_constant}')

