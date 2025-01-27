import matplotlib.pyplot as plt
import pickle
import os
from matplotlib.ticker import ScalarFormatter

#### CREIAMO E SALVIAMO I PLOT DELLE PREDICTION SUL TEST-SET ####
#### CREIAMO E SALVIAMO I PLOT DELLE ROC SUL TEST-SET ####

archs=['bottle','funnel','constant']


for arch in archs:
    path='/home/private/Hepd/Dataset_4/fcNN/PCA/'+arch
    model_list=os.listdir(path)


    
    for i,model in enumerate(model_list):
        print('We are plotting the model '+model)
        print(f'It is the number {i+1} over a total of {len(model_list)}')

        #### APRIAMO LE PREDICTIONS ###
        with open(path+'/'+model+'/'+model+'_electrons_predictions.pkl','rb') as f:
            electrons_predictions=pickle.load(f)
        with open(path+'/'+model+'/'+model+'_protons_prediction.pkl','rb') as f:
            protons_predictions=pickle.load(f)
    
    ### Plottiamo le predictions per Elettroni e Protoni ###
        fig, ax=plt.subplots()
        ax.hist(electrons_predictions,bins=100,color="blue",label="Electrons")
        ax.hist(protons_predictions,bins=100,color="red",label="Protons",alpha=0.5)
        ax.legend()
        plt.xlabel('Prediction')
        plt.ylabel('Frequency')
        plt.yscale('log')
        plt.title('Predictions model PCA '+model+' over Test set')
        plt.savefig(path+'/'+model+'/'+model+'_plot_test_predictions.png')
        plt.close()

        
        #### APRIAMO LE FPR E TPR E LA ROC_AUC ###
        
        with open(path+'/'+model+'/'+model+'_test_fpr.pkl','rb') as f:
            fpr=pickle.load(f)
        
        with open(path+'/'+model+'/'+model+'_test_tpr.pkl','rb') as f:
            tpr=pickle.load(f)
    
        with open(path+'/'+model+'/'+model+'_test_ROC_auc.pkl','rb') as f:
            roc_auc=pickle.load(f)    

        
        ### PLottiamo le ROC e salviamo in un file il valore dell'area sottesa ###
        plt.figure()
        lw = 2
        plt.plot(tpr,1-fpr, color='green', lw=lw, label='ROC curve (area = %0.4f)' % roc_auc)
        plt.minorticks_on()
        plt.gca().yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        plt.xlabel('Signal Efficiency')
        plt.ylabel('Background Rejection')
        plt.title('ROC model PCA '+model+' over Test set')
        plt.legend()
        plt.savefig(path+'/'+model+'/'+model+'_plot_test_ROC.png')
        plt.close()

        #### APRIAMO IL FILE HISTORY PER OTTENERE LE LOSS E ACCURACY ####
    
        with open(path+'/'+model+'/'+model+'_history.pkl','rb') as f:
            metrics=pickle.load(f)

        #Otteniamo le loss da plottare
        train_loss=metrics['loss']
        val_loss=metrics['val_loss']
    
        plt.figure(figsize=(10, 6))
        plt.plot(train_loss,color='red', label='Training Loss')
        plt.plot(val_loss,color='blue', label='Validation Loss')
        plt.title('Training and Validation Loss model PCA '+model)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(path+'/'+model+'/'+model+'_loss.png')
        plt.close()

        # Otteniamo le accuracy da plottare
        train_accuracy=metrics['accuracy']
        val_accuracy=metrics['val_accuracy']
    
        plt.figure(figsize=(10, 6))
        plt.plot(train_accuracy,color='red', label='Training Accuracy')
        plt.plot(val_accuracy,color='blue', label='Validation Accuracy')
        plt.title('Training and Validation Accuracy model PCA '+model)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(path+'/'+model+'/'+model+'_accuracy.png')
        plt.close()
        
    print('Done plotting the '+arch+' models \n')
    
print('Done')