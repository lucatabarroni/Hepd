import lightgbm as lgb
import os
import pickle
import joblib
from tqdm import tqdm
import warnings

### il learning rate è il medesimo per tutti i tree
learning_rate = 0.05
### il tipo di boosting è il medesimo per tutti i tree
boost_type = 'gbdt'
### la loss è la medesima per tutti gli ensembles
loss = 'binary_logloss'

#### definiamo un intero train set 
#### N.B. qui non abiamo un validation set che viene valutato durante il training
#### per le BDT utilizziamo il dataset non normalizzato
list_batch_data=os.listdir('/home/private/Hepd/Dataset_4/train/train_data/')
batch_data=[]
batch_labels=[]
for i in range(len(list_batch_data)):
    with open('/home/private/Hepd/Dataset_4/train/train_data/'+str(i)+'.pkl','rb') as f:
        batch_data.extend(pickle.load(f))
    with open('/home/private/Hepd/Dataset_4/train/train_labels/'+str(i)+'.pkl','rb') as f:
        batch_labels.extend(pickle.load(f))

# Definisci un callback per aggiornare la barra di avanzamento
class TQDMProgressBar:
    def __init__(self, total_iterations):
        self.total_iterations = total_iterations
        self.progress_bar = None

    def __call__(self, env):
        if self.progress_bar is None:
            self.progress_bar = tqdm(total=self.total_iterations, desc='Training Progress')
        self.progress_bar.update(1)
        if env.iteration + 1 == self.total_iterations:
            self.progress_bar.close()
            
### otteniamo la lista di tutti i modelli da trainare con i relativi path

### in path names ci sono tutte le cartelle con i nomi dei modelli da trainare
### il formato del nome è model_name = n_estimators_N1_max_depth_N2_n_leaves_N3
path_names = '/home/private/Hepd/Dataset_4/bdt'

### in list models ci sono tutti i nomi dei modelli
list_models = os.listdir(path_names)

### in ogni cartella c'è un file model_name.summary 
### N. estimators : N1
### Max depth : N2
### N. leaves : N3
### Min data in leaf : N4

### otteniamo i path delle cartelle in cui andreamo a lavorare
list_dir_paths = [path_names+'/'+model_name+'/'+model_name for model_name in list_models]

### leggiamo quindi i valori numerici alla fine di ogni riga
i = 0
for k,dir_path in enumerate(list_dir_paths) :
    print(f'stiamo trainando la bdt {dir_path}, è il numero {k+1} su {len(list_dir_paths)}')
    summary_path = dir_path+'_summary.txt'
    with open(summary_path,'r') as f:
        # otteniamo tutte le righe
        lines = f.readlines()
        ### l'ultimo numero alla fine della prima riga è n_estimators
        ### l'ultimo alla dine della seconda è max_depth
        ### l'ultimo alla fine della terza è n_leaves
        for i,line in enumerate(lines):
            if i == 0:
                n_estimators = int(line.split()[-1])
            if i == 1:
                max_depth = int(line.split()[-1])
            if i == 2:
                n_leaves = int(line.split()[-1])
            if i == 3:
                min_data_leaf = int(line.split()[-1])
    if n_estimators == -1:
        continue
    ### andiamo a creare una bdt con i parametri considerati
    ### mettendo verbose = -1 eliminiamo i warning
    ### alcuni bdt potrebbero avere il warning: [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    ### probabilmente deriva da valori di min_data_in_leaf troppo grandi
    gbm = lgb.LGBMClassifier(boosting_type = boost_type, num_leaves = n_leaves, max_depth = max_depth, min_child_samples = min_data_leaf, learning_rate = learning_rate, n_estimators=n_estimators, objective='binary', metric = loss,verbose = -1)
    ### istanziamo il callback per avere la barra di avanzamento del fit
    progress_bar_callback = TQDMProgressBar(total_iterations=n_estimators)
    ### eseguiamo il fit
    gbm.fit(batch_data,batch_labels, callbacks = [progress_bar_callback])
    ### definiamo il percorso del file in cui salvare il modello trainato e ce lo salviamo
    model_path = dir_path+'_trained.pkl'
    joblib.dump(gbm, model_path)