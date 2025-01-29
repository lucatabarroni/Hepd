from sklearn.preprocessing import StandardScaler
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
import seaborn as sns
from matplotlib.colors import LogNorm

### FACCIAMO DELLA PCA SUI DATI DI HEPD ###

#### RENDIAMO I DATI DI TRAIN NORMALIZZATI CON MEDIA ZERO E VARIANZA 1 ####

### DEFINIAMO LO SCALER CHE DI DEFAULT HA MEDIA ZERO E VARIANZA 1 ###

scaler=StandardScaler()

### definiamo la lista dei percorsi per i dati di train

train_dir='/home/private/Hepd/Dataset_4/train/train_norm_data/'
train_data_list=[train_dir+batch_num for batch_num in os.listdir(train_dir)]

### definiamo la lista dei percorsi per il test set

test_dir='/home/private/Hepd/Dataset_4/test/test_norm_data/'
test_data_list=[test_dir+batch_num for batch_num in os.listdir(test_dir)]

### definiamo la funzione per il fit di scaler su un batch di dati

def scaler_fit_batch(scaler,batch):
    return scaler.partial_fit(batch)

### invochiamo la funzione per il fit a batch per tutti i batch del train set ###

for batch_path in train_data_list:
    with open(batch_path,'rb') as f:
        batch=pickle.load(f)
    scaler_fit_batch(scaler,batch)

#### salviamo lo scaler fittato ####

with open('scaler.pkl','wb') as f:
    pickle.dump(scaler,f)

### salviamo tutti i dati di train in un nuovo Pandas dataframe

train_data_transform=[]
for i in range(len(train_data_list)):
    with open(train_data_list[i],'rb') as f:
        data=pickle.load(f)
    train_data_transform.extend(scaler.transform(data))

df_train_data_transform=pd.DataFrame(train_data_transform)

### salviamo i dati di test in un nuovo dataframe

test_data_transform=[]
for i in range(len(test_data_list)):
    with open(test_data_list[i],'rb') as f:
        data=pickle.load(f)
    test_data_transform.extend(scaler.transform(data))

df_test_data_transform=pd.DataFrame(test_data_transform)

### plottiamo le nuove features riscalate 

fig, axes = plt.subplots(8, 5, figsize=(20, 20))
for i,col in enumerate(df_train_data_transform.columns):
    row = i // 5
    col_idx = i % 5
    df_train_data_transform[col].hist(ax=axes[row, col_idx],bins=50,label='Train set')
    df_test_data_transform[col].hist(ax=axes[row, col_idx],bins=50,color='red',alpha=0.5,label='Test set')
    axes[row, col_idx].set_title(col)
    axes[row, col_idx].legend()

# Aggiusta il layout per evitare sovrapposizioni
plt.tight_layout()

# Salva la figura in un file
plt.title('Rescaled features distribution')
plt.savefig('rescaled_features_plots.png')

plt.show()


#### PLOTTIAMO LA CORRELAZIONE TRA LE VARIABILI (LA VARIANZA è UNO) PER IL TRAIN SET ####

train_covariancy=df_train_data_transform.cov()

plt.figure(figsize=(8, 6))
sns.heatmap(train_covariancy, cmap='coolwarm')
plt.title('Rescaled train features correlation')
plt.savefig('rescaled_train_correlation.png')
plt.show()


#### PLOTTIAMO LA CORRELAZIONE TRA LE VARIABILI (LA VARIANZA è UNO) PER IL TRAIN SET ####

test_covariancy=df_test_data_transform.cov()

plt.figure(figsize=(8, 6))
sns.heatmap(test_covariancy, cmap='coolwarm')
plt.title('Rescaled test features correlation')
plt.savefig('rescaled_test_correlation.png')
plt.show()

### DEFINIAMO L'OGGETTO PCA 

pca=PCA()

### FITTIAMO L'OGGETTO PCA SU TUTTO IL DATAFRAME RISCALATO 

pca.fit(df_train_data_transform)

#### Salviamo la trasformazione pca
with open('pca_transform.pkl','wb') as f:
    pickle.dump(pca,f)

# Ottieniamo la matrice del cambio di base (componenti principali)
components = pca.components_

# Salviamo la matrice in un file
with open('pca_components.pkl','wb') as f:
    pickle.dump(components,f)

### VALUTIAMO IL DATASET NELLA NUOVA BASE OTTENUTA DALLA PCA

df_train_data_pca=pd.DataFrame(pca.transform(df_train_data_transform))

with open('train.pkl','wb') as f:
    pickle.dump(df_train_data_transform,f)


with open('train_pca.pkl','wb') as f:
    pickle.dump(df_train_data_pca,f)

df_test_data_pca=pd.DataFrame(pca.transform(df_test_data_transform))

### plottiamo le feature in questa nuova base 

fig, axes = plt.subplots(8, 5, figsize=(20, 20))
for i,col in enumerate(df_train_data_pca.columns):
    row = i // 5
    col_idx = i % 5
    df_train_data_pca[col].hist(ax=axes[row, col_idx],bins=50,color='b',label='Train set')
    df_test_data_pca[col].hist(ax=axes[row, col_idx],bins=50,color='red',alpha=0.5,label='Test set')
    axes[row, col_idx].set_title(col)
    axes[row, col_idx].legend()

# Aggiusta il layout per evitare sovrapposizioni
plt.tight_layout()

# Salva la figura in un file
plt.title('PCA features')
plt.savefig('pca_features_plots.png')

plt.show()

### VALUTIAMO LA CORRELAZIONE TRA LE VARIABILI NELLA NUOVA BASE PCA

pca_train_covariancy=df_train_data_pca.cov()

### PLOTTIAMO LA CORRELAZIONE TRA LE VARIABILI NELLA NUOVA BASE

plt.figure(figsize=(8, 6))
sns.heatmap(pca_train_covariancy,  cmap='coolwarm')
plt.title('PCA features train correlation')
plt.savefig('pca_train_correlation.png')
plt.show()

### PLOTTIAMO LA CORRELAZIONE NELLA NUOVA BASE IN SCALA LOGARITMINCA

plt.figure(figsize=(8, 6))
sns.heatmap(pca_train_covariancy,  cmap='coolwarm', norm=LogNorm())
plt.title('PCA features train correlation logaritmic scale')
plt.savefig('pca_train_correlation_LOG.png')
plt.show()


### VALUTIAMO LA CORRELAZIONE TRA LE VARIABILI NELLA NUOVA BASE PCA DEL TEST

pca_test_covariancy=df_test_data_pca.cov()

### PLOTTIAMO LA CORRELAZIONE TRA LE VARIABILI NELLA NUOVA BASE

plt.figure(figsize=(8, 6))
sns.heatmap(pca_test_covariancy,  cmap='coolwarm')
plt.title('PCA features test correlation')
plt.savefig('pca_test_correlation.png')
plt.show()

### PLOTTIAMO LA CORRELAZIONE NELLA NUOVA BASE IN SCALA LOGARITMINCA

plt.figure(figsize=(8, 6))
sns.heatmap(pca_test_covariancy,  cmap='coolwarm', norm=LogNorm())
plt.title('PCA features test correlation logaritmic scale')
plt.savefig('pca_test_correlation_LOG.png')
plt.show()
