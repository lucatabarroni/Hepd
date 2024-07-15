import os
import numpy as np
import keras
import pickle
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
from keras.models import load_model

import tensorflow as tf
tf.config.list_physical_devices('GPU')

class MyGenerator(keras.utils.Sequence):
    def __init__(self, data_dir, labels_dir, batch_size=1):
        self.data_dir = data_dir
        self.labels_dir = labels_dir
        self.batch_size = batch_size
        self.files = os.listdir(data_dir)
        self.indexes = np.arange(len(self.files))

    def __len__(self):
        return int(len(self.files)*(4096/self.batch_size)) 

    def __getitem__(self, index):
        with open(self.data_dir+str(index)+'.pkl','rb') as f:
            batch_data=pickle.load(f)

        with open(self.labels_dir+str(index)+'.pkl','rb') as f:
            batch_labels=pickle.load(f)
        
        return np.array(batch_data), np.array(batch_labels)

test_generator = MyGenerator(data_dir='/home/private/Hepd/Dataset_3/batched_dataset/test/bs_4096/data/', labels_dir='/home/private/Hepd/Dataset_3/batched_dataset/test/bs_4096/labels/', batch_size=4096)


# In[10]:


type=''
parameters=[]
for num_par in parameters:
    path='/home/private/Hepd/'+type+'_model_pre-trained/'+num_par+'/trained_'+type+num_par+'.h5'

    model=load_model(path)
    predictions = []
    labels = []
    for i in range(len(test_generator)):
        data, label = test_generator[i]
        pred = model.predict(data,batch_size=4096,verbose=0)
        predictions.extend(pred.flatten())
        labels.extend(label.flatten())
    
    
    # In[14]:
    
    
    fpr, tpr, thresholds = roc_curve(labels, predictions)
    
    efficiency = tpr
    background_rejection = 1 - fpr
    
    area = auc(efficiency, background_rejection)
    
    # Disegna il grafico
    plt.figure()
    plt.plot(efficiency, background_rejection, label='Area = %0.6f' % area)
    plt.xlim([0.0, 1.01])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Signal Efficiency')
    plt.ylabel('Background Rejection')
    plt.title('ROC '+type+' '+num_par)
    plt.legend(loc="lower right")
    plt.savefig('ROC_'+type+'_'+num_par+'.png')
    plt.show()
    
    
    # In[16]:
    
    
    # Salva i dati in un file di testo
    with open('roc_area.txt', 'a') as f:
        f.write(f"{type}, {num_par}, {area}\n")
    
    import pickle
    
    # Salva predictions e labels in un file pickle
    with open(type+'_'+num_par+'.pkl', 'wb') as f:
        pickle.dump((predictions, labels), f)