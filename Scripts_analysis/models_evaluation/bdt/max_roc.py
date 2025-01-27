import pickle
import numpy as np
import os

#### vogliamo leggere le aree delle ROC dei modelli bdt
### le aree sono salvati in due file txt:
### - uno per la ROC ottenuta con la sigmoide del raw score,
### - uno per la roc ottenuta con la probabilità della classe signal assegnata all'evento dalla bdt
### I file sono salvati con il formato
### {nome_modello}_ROC_area_predictions_sigmoid.txt
### {nome_modello}_ROC_area_probabilities_1.txt
### dento entrambi i file c'è una riga di testo che si conclude con il valore dell'area della roc


### definiamo una funzione che prenda il percorso del file roc_area ed estragga il valore dall'ultima riga
def get_roc_area (roc_area_file_path):
    with open(roc_area_file_path, 'r') as f:
        ### eliminiamo eventuali spazi e andate a capo
        first_line = f.readline().strip()
        ### il valore dell'area è alla fine quindi invertiamo la riga
        line = first_line[::-1]
        ### troviamo il primo spazio
        whitespace_index = line.find(' ')
        ### trasformiamo il valore in un float
        roc_area = float(line[:whitespace_index][::-1])
        return roc_area

### creiamo la lista con tutti i nomi dei modelli di cui estrarre le ROC
path = '/home/private/Hepd/Dataset_4/bdt'
model_list = os.listdir(path)

max_predictions_area = -1
max_probabilities_area = -1
max_predictions_index = -1
max_probabilities_index = -1


for i,model_name in enumerate(model_list):
    ### definiamo i path dei ROC files
    predictions_roc_area_path = path+'/'+model_name+'/'+model_name+'_ROC_area_predictions_sigmoid.txt'
    probabilities_roc_area_path = path+'/'+model_name+'/'+model_name+'_ROC_area_probabilities_1.txt'
    ### otteniamo le due roc area
    roc_area_predictions = get_roc_area(predictions_roc_area_path)
    roc_area_probabilities_1 = get_roc_area(probabilities_roc_area_path)
    ### salviamo i valori delle aree in un file insieme al modello da cui arrivano
    with open('/home/private/Hepd/Dataset_4/Evaluation/bdt/rocs_predictions_sigmoid.txt', 'a') as f:
        f.write(f'{model_name}: {roc_area_predictions}\n')
    with open('/home/private/Hepd/Dataset_4/Evaluation/bdt/rocs_probabilities_1.txt', 'a') as f:
        f.write(f'{model_name}: {roc_area_probabilities_1}\n')
    ### controlliamo se i valori sono più grandi dei massimi che abbiamo fino ad ora
    if roc_area_predictions >= max_predictions_area:
        max_predictions_area = roc_area_predictions
        max_predictions_index = i
    if roc_area_probabilities_1 >= max_probabilities_area:
        max_probabilities_area = roc_area_probabilities_1
        max_probabilities_index = i
        
### una volta controllate tutte le aree ci salviamo i valori massimi in due file
with open('/home/private/Hepd/Dataset_4/Evaluation/bdt/best_roc_predictions_sigmoid.txt', 'w') as f:
        f.write(f'{model_list[max_predictions_index]}: {max_predictions_area}\n')
with open('/home/private/Hepd/Dataset_4/Evaluation/bdt/best_roc_probabilities_1.txt', 'w') as f:
        f.write(f'{model_list[max_probabilities_index]}: {max_probabilities_area}\n')















    