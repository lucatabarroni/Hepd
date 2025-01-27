import os
import matplotlib.pyplot as plt
import pickle


#### CREIAMO I PLOT DELLE LOSS AL VARIARE DI PROFONDITà E LARGHEZZA DI PRIMO E ULTIMO LAYER PER I BOTTLE MODELS ####

list_bottle_models=os.listdir('/home/private/Hepd/Dataset_4/fcNN/PCA/bottle')
arch='bottle'


#### CREIAMO UNA LISTA DI INTERI CON LE POSSIBILI LARGHEZZE DELL'ULTIMO/PRIMO LAYER ####
layers=[]
for model in list_bottle_models:
    layer='empty'
    #### invertiamo il nome del modello e prendiamo il primo numero che si riferisce al 
    #### primo/ultimo layer
    for i,c in enumerate(model[::-1]):
        if c!='_':
            continue
        else:
            layer=model[::-1][:i]
            break
    if int(layer[::-1]) not in layers:
        layers.append(int(layer[::-1]))

#### CREIAMO UNA LISTA DI INTERI CON LE POSSIBILI DEPTHS ####
depths=[]
#### LA DEPTH VIENE PRIMA DELL'INDICAZIONE SULL'ULTIMO7PRIMO LAYER,
#### QUINDI CI SERVE QUALCOSA PER CONTARE I SEPARATORI
starting_depth=0
for model in list_bottle_models:
    underscore_counter=0
    depth='empty'
    for i,c in enumerate(model[::-1]):
        if c!='_':
            continue
        elif underscore_counter != 1:
            starting_depth=i
            underscore_counter +=1
            continue
        else:
            depth=model[::-1][starting_depth+1:i]
            break
    if int(depth[::-1]) not in depths:
        depths.append(int(depth[::-1]))
depths.sort()
layers.sort()

#### PLOTTIAMO LE ROC E LE LOSS PER LAYER COSTANTE ####
#### DA QUESTA DIR ANDREMO A PRENDERE I VALORI DA PLOTTARE ####
prel_dir='/home/private/Hepd/Dataset_4/fcNN/PCA/bottle/bottle_PCA_depth_'
rocs=[]
train_loss=[]
validation_loss=[]
#### CICLIAMO SU TUTTE LE DEPTHS ####
for depth in depths:
    rocs=[]
    train_loss=[]
    validation_loss=[]
    test_loss=[]
    #### PER OGNI DEPTH CICLIAMO SU TUTTE LE LARGHEZZE DELL'ULTIMO LAYER ####
    for layer in layers:
        
        model=arch+'_PCA_depth_'+str(depth)+'_'+str(layer)
        #### COMPLETIAMO IL PERCORSO DELLA DIRECTORY DA QUI PRENDIAMO I DATI CON LA DEFINIZIONE DEL MODELLO ####
        prel_dir_fin=prel_dir+str(depth)+'_'+str(layer)
        #### PER UNA PROFONDITà DI 20 NON ABBIAMO IL MODELLO CON ULTIMO LAYER LARGO 32 ####
        if depth==20 and layer==32:
            continue
        else:    
            with open(prel_dir_fin+'/'+model+'_test_loss_accuracy.txt','r') as f:
                f.readline()
                te_loss = float(f.readline().strip()[6:])
            test_loss.append(te_loss)

            with open(prel_dir_fin+'/'+model+'_train_loss_accuracy.txt','r') as f:
                f.readline()
                tr_loss = float(f.readline().strip()[6:])
            train_loss.append(tr_loss)

            with open(prel_dir_fin+'/'+model+'_validation_loss_accuracy.txt','r') as f:
                f.readline()
                val_loss = float(f.readline().strip()[6:])
            validation_loss.append(val_loss)

            with open(prel_dir_fin+'/'+model+'_test_ROC_auc.pkl','rb') as f:
                roc=pickle.load(f)
                rocs.append(roc)
    #### PLOTTIAMO I RISULTATI PER OGNI DEPTH ####
    plt.figure(figsize=(14, 8))
    plt.grid(True)
    if depth==20:
        plt.plot(layers[1:],train_loss,color='r',label='Train Loss')
        plt.plot(layers[1:],validation_loss,color='b',label='Validation Loss',alpha=0.7)
        plt.plot(layers[1:],test_loss,color='g',label='Test Loss',alpha=0.4)
        plt.xticks(layers[1:])
    else:
        plt.plot(layers,train_loss,color='r',label='Train Loss')
        plt.plot(layers,validation_loss,color='b',label='Validation Loss',alpha=0.7)
        plt.plot(layers,test_loss,color='g',label='Test Loss',alpha=0.4)
        plt.xticks(layers)
    plt.xlabel('Last Layer Width')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.title('Train Test Validation Loss with depth '+str(depth)+' of PCA bottle models')
    plt.savefig('/home/private/Hepd/Dataset_4/analysis/PCA/bottle/losses/depths/losses_PCA_depth_'+str(depth)+'.png')
    plt.close()

    plt.figure(figsize=(14, 8))
    plt.grid(True)
    if depth==20:
        plt.plot(layers[1:],rocs,color='g',label='Test ROC area')
        plt.xticks(layers[1:])
    else:
        plt.plot(layers,rocs,color='g',label='Test ROC area')
        plt.xticks(layers)
    plt.xlabel('Last Layer Width')
    plt.ylabel('ROC Value')
    plt.legend()
    plt.title('Test ROCs with depth '+str(depth)+' of PCA bottle models')
    plt.savefig('/home/private/Hepd/Dataset_4/analysis/PCA/bottle/rocs/depths/rocs_PCA_depth_'+str(depth)+'.png')
    plt.close()


#### PLOTTIAMO LE ROC E LE LOSS PER DEPTH COSTANTE ####
prel_dir='/home/private/Hepd/Dataset_4/fcNN/PCA/bottle/bottle_PCA_depth_'
rocs=[]
train_loss=[]
validation_loss=[]
#### CICLIAMO SU TUTTE LE LARGHEZZE DELL'ULTIMO LAYER ####
for layer in layers:
    rocs=[]
    train_loss=[]
    validation_loss=[]
    test_loss=[]
    #### PER OGNI LARGHEZZA DELL'ULTIMO LAYER CICLIAMO SU TUTTE LE DEPTH ####
    for depth in depths:
        model=arch+'_PCA_depth_'+str(depth)+'_'+str(layer)
        prel_dir_fin=prel_dir+str(depth)+'_'+str(layer)
        if depth==20 and layer==32:
            continue
        else:    
            with open(prel_dir_fin+'/'+model+'_test_loss_accuracy.txt','r') as f:
                f.readline()
                te_loss = float(f.readline().strip()[6:])
            test_loss.append(te_loss)

            with open(prel_dir_fin+'/'+model+'_train_loss_accuracy.txt','r') as f:
                f.readline()
                tr_loss = float(f.readline().strip()[6:])
            train_loss.append(tr_loss)

            with open(prel_dir_fin+'/'+model+'_validation_loss_accuracy.txt','r') as f:
                f.readline()
                val_loss = float(f.readline().strip()[6:])
            validation_loss.append(val_loss)

            with open(prel_dir_fin+'/'+model+'_test_ROC_auc.pkl','rb') as f:
                roc=pickle.load(f)
                rocs.append(roc)
    plt.figure(figsize=(14, 8))
    plt.grid(True)
    if layer==32:
        plt.plot(depths[:-1],train_loss,color='r',label='Train Loss')
        plt.plot(depths[:-1],validation_loss,color='b',label='Validation Loss',alpha=0.7)
        plt.plot(depths[:-1],test_loss,color='g',label='Test Loss',alpha=0.4)
        plt.xticks(depths[:-1])
    else:
        plt.plot(depths,train_loss,color='r',label='Train Loss')
        plt.plot(depths,validation_loss,color='b',label='Validation Loss',alpha=0.7)
        plt.plot(depths,test_loss,color='g',label='Test Loss',alpha=0.4)
        plt.xticks(depths)
    plt.xlabel('Depth')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.title('Train Test Validation Loss with last layer of '+str(layer)+' neurons of PCA bottle models')
    plt.savefig('/home/private/Hepd/Dataset_4/analysis/PCA/bottle/losses/layers/losses_PCA_layer_'+str(layer)+'.png')
    plt.close()

    plt.figure(figsize=(14, 8))
    plt.grid(True)
    if layer==32:
        plt.plot(depths[:-1],rocs,color='g',label='Test ROC area')
        plt.xticks(depths[:-1])
    else:
        plt.plot(depths,rocs,color='g',label='Test ROC area')
        plt.xticks(depths)
    plt.xlabel('Depth')
    plt.ylabel('ROC Value')
    plt.legend()
    plt.title('Test ROCs with last layer of '+str(layer)+' neurons of PCA bottle models')
    plt.savefig('/home/private/Hepd/Dataset_4/analysis/PCA/bottle/rocs/layers/rocs_PCA_layer_'+str(layer)+'.png')
    plt.close()



#### CREIAMO I PLOT DELLE LOSS AL VARIARE DI PROFONDITà E LARGHEZZA DI PRIMO E ULTIMO LAYER DEI FUNNEL LAYERS ####

list_funnel_models=os.listdir('/home/private/Hepd/Dataset_4/fcNN/PCA/funnel/')
arch='funnel'

#### CREIAMO UNA LISTA DI INTERI CON LE POSSIBILI LARGHEZZE DELL'ULTIMO/PRIMO LAYER ####
layers=[]
for model in list_funnel_models:
    layer='empty'
    #### invertiamo il nome del modello e prendiamo il primo numero che si riferisce al 
    #### primo/ultimo layer
    for i,c in enumerate(model[::-1]):
        if c!='_':
            continue
        else:
            layer=model[::-1][:i]
            break
    if int(layer[::-1]) not in layers:
        layers.append(int(layer[::-1]))

#### CREIAMO UNA LISTA DI INTERI CON LE POSSIBILI DEPTHS ####
depths=[]
#### LA DEPTH VIENE PRIMA DELL'INDICAZIONE SULL'ULTIMO7PRIMO LAYER,
#### QUINDI CI SERVE QUALCOSA PER CONTARE I SEPARATORI
starting_depth=0
for model in list_funnel_models:
    underscore_counter=0
    depth='empty'
    for i,c in enumerate(model[::-1]):
        if c!='_':
            continue
        elif underscore_counter != 1:
            starting_depth=i
            underscore_counter +=1
            continue
        else:
            depth=model[::-1][starting_depth+1:i]
            break
    if int(depth[::-1]) not in depths:
        depths.append(int(depth[::-1]))
depths.sort()
layers.sort()

#### PLOTTIAMO LE ROC E LE LOSS PER LAYER COSTANTE ####
prel_dir='/home/private/Hepd/Dataset_4/fcNN/PCA/funnel/funnel_PCA_depth_'
rocs=[]
train_loss=[]
validation_loss=[]
for depth in depths:
    rocs=[]
    train_loss=[]
    validation_loss=[]
    test_loss=[]
    for layer in layers:
        layers_to_plot=[]
        model=arch+'_PCA_depth_'+str(depth)+'_'+str(layer)
        prel_dir_fin=prel_dir+str(depth)+'_'+str(layer)
        if depth==20 and layer==32:
            continue
        else:    
            with open(prel_dir_fin+'/'+model+'_test_loss_accuracy.txt','r') as f:
                f.readline()
                te_loss = float(f.readline().strip()[6:])
            test_loss.append(te_loss)

            with open(prel_dir_fin+'/'+model+'_train_loss_accuracy.txt','r') as f:
                f.readline()
                tr_loss = float(f.readline().strip()[6:])
            train_loss.append(tr_loss)

            with open(prel_dir_fin+'/'+model+'_validation_loss_accuracy.txt','r') as f:
                f.readline()
                val_loss = float(f.readline().strip()[6:])
            validation_loss.append(val_loss)

            with open(prel_dir_fin+'/'+model+'_test_ROC_auc.pkl','rb') as f:
                roc=pickle.load(f)
                rocs.append(roc)
    plt.figure(figsize=(14, 8))
    plt.grid(True)
    if depth==20:
        plt.plot(layers[1:],train_loss,color='r',label='Train Loss')
        plt.plot(layers[1:],validation_loss,color='b',label='Validation Loss',alpha=0.7)
        plt.plot(layers[1:],test_loss,color='g',label='Test Loss',alpha=0.4)
        plt.xticks(layers[1:])
    else:
        plt.plot(layers,train_loss,color='r',label='Train Loss')
        plt.plot(layers,validation_loss,color='b',label='Validation Loss',alpha=0.7)
        plt.plot(layers,test_loss,color='g',label='Test Loss',alpha=0.4)
        plt.xticks(layers)
    plt.xlabel('First Layer Width')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.title('Train Test Validation Loss with depth '+str(depth)+' of funnel PCA models')
    plt.savefig('/home/private/Hepd/Dataset_4/analysis/PCA/funnel/losses/depths/losses_PCA_depth_'+str(depth)+'.png')
    plt.close()

    plt.figure(figsize=(14, 8))
    plt.grid(True)
    if depth==20:
        plt.plot(layers[1:],rocs,color='g',label='Test ROC area')
        plt.xticks(layers[1:])
    else:
        plt.plot(layers,rocs,color='g',label='Test ROC area')
        plt.xticks(layers)
    plt.xlabel('First Layer Width')
    plt.ylabel('ROC Value')
    plt.legend()
    plt.title('Test ROCs with depth '+str(depth)+' of PCA funnel models')
    plt.savefig('/home/private/Hepd/Dataset_4/analysis/PCA/funnel/rocs/depths/rocs_PCA_depth_'+str(depth)+'.png')
    plt.close()


#### PLOTTIAMO LE ROC E LE LOSS PER DEPTH COSTANTE ####
prel_dir='/home/private/Hepd/Dataset_4/fcNN/PCA/funnel/funnel_PCA_depth_'
arch='funnel'
rocs=[]
train_loss=[]
validation_loss=[]
for layer in layers:
    rocs=[]
    train_loss=[]
    validation_loss=[]
    test_loss=[]
    for depth in depths:
        model=arch+'_PCA_depth_'+str(depth)+'_'+str(layer)
        prel_dir_fin=prel_dir+str(depth)+'_'+str(layer)
        if depth==20 and layer==32:
            continue
        else:    
            with open(prel_dir_fin+'/'+model+'_test_loss_accuracy.txt','r') as f:
                f.readline()
                te_loss = float(f.readline().strip()[6:])
            test_loss.append(te_loss)

            with open(prel_dir_fin+'/'+model+'_train_loss_accuracy.txt','r') as f:
                f.readline()
                tr_loss = float(f.readline().strip()[6:])
            train_loss.append(tr_loss)

            with open(prel_dir_fin+'/'+model+'_validation_loss_accuracy.txt','r') as f:
                f.readline()
                val_loss = float(f.readline().strip()[6:])
            validation_loss.append(val_loss)

            with open(prel_dir_fin+'/'+model+'_test_ROC_auc.pkl','rb') as f:
                roc=pickle.load(f)
                rocs.append(roc)
    plt.figure(figsize=(14, 8))
    plt.grid(True)
    if layer==32:
        plt.plot(depths[:-1],train_loss,color='r',label='Train Loss')
        plt.plot(depths[:-1],validation_loss,color='b',label='Validation Loss',alpha=0.7)
        plt.plot(depths[:-1],test_loss,color='g',label='Test Loss',alpha=0.4)
        plt.xticks(depths[:-1])
    else:
        plt.plot(depths,train_loss,color='r',label='Train Loss')
        plt.plot(depths,validation_loss,color='b',label='Validation Loss',alpha=0.7)
        plt.plot(depths,test_loss,color='g',label='Test Loss',alpha=0.4)
        plt.xticks(depths)
    plt.xlabel('Depth')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.title('Train Test Validation Loss with first layer of '+str(layer)+' neurons of PCA funnel models')
    plt.savefig('/home/private/Hepd/Dataset_4/analysis/PCA/funnel/losses/layers/losses_PCA_layer_'+str(layer)+'.png')
    plt.close()

    plt.figure(figsize=(14, 8))
    plt.grid(True)
    if layer==32:
        plt.plot(depths[:-1],rocs,color='g',label='Test ROC area')
        plt.xticks(depths[:-1])
    else:
        plt.plot(depths,rocs,color='g',label='Test ROC area')
        plt.xticks(depths)
    plt.xlabel('Depth')
    plt.ylabel('ROC Value')
    plt.legend()
    plt.title('Test ROCs with first layer of '+str(layer)+' neurons of PCA funnel models')
    plt.savefig('/home/private/Hepd/Dataset_4/analysis/funnel/rocs/layers/rocs_PCA_layer_'+str(layer)+'.png')
    plt.close()


#### CREIAMO I PLOT DELLE LOSS AL VARIARE DI PROFONDITà E LARGHEZZA DI PRIMO E ULTIMO LAYER DEI CONSTANT MODELS ####

list_constant_models=os.listdir('/home/private/Hepd/Dataset_4/fcNN/PCA/constant/')
arch='constant'

#### CREIAMO UNA LISTA DI INTERI CON LE POSSIBILI LARGHEZZE DELL'ULTIMO/PRIMO LAYER ####
layers=[]
for model in list_constant_models:
    layer='empty'
    #### invertiamo il nome del modello e prendiamo il primo numero che si riferisce al 
    #### primo/ultimo layer
    for i,c in enumerate(model[::-1]):
        if c!='_':
            continue
        else:
            layer=model[::-1][:i]
            break
    if int(layer[::-1]) not in layers:
        layers.append(int(layer[::-1]))

#### CREIAMO UNA LISTA DI INTERI CON LE POSSIBILI DEPTHS ####
depths=[]
#### LA DEPTH VIENE PRIMA DELL'INDICAZIONE SULL'ULTIMO7PRIMO LAYER,
#### QUINDI CI SERVE QUALCOSA PER CONTARE I SEPARATORI
starting_depth=0
for model in list_constant_models:
    underscore_counter=0
    depth='empty'
    for i,c in enumerate(model[::-1]):
        if c!='_':
            continue
        elif underscore_counter != 1:
            starting_depth=i
            underscore_counter +=1
            continue
        else:
            depth=model[::-1][starting_depth+1:i]
            break
    if int(depth[::-1]) not in depths:
        depths.append(int(depth[::-1]))
depths.sort()
layers.sort()

#### PLOTTIAMO LE ROC E LE LOSS PER LAYER COSTANTE ####
prel_dir='/home/private/Hepd/Dataset_4/fcNN/PCA/constant/constant_PCA_depth_'
rocs=[]
train_loss=[]
validation_loss=[]
for depth in depths:
    rocs=[]
    train_loss=[]
    validation_loss=[]
    test_loss=[]
    for layer in layers:
        layers_to_plot=[]
        model=arch+'_PCA_depth_'+str(depth)+'_'+str(layer)
        prel_dir_fin=prel_dir+str(depth)+'_'+str(layer)
        #### N.B. IN QUESTO CASO ESISTE IL MODELLO CON DEPTH 20 E ULTIMO LAYER DA 32 ####
        with open(prel_dir_fin+'/'+model+'_test_loss_accuracy.txt','r') as f:
            f.readline()
            te_loss = float(f.readline().strip()[6:])
        test_loss.append(te_loss)

        with open(prel_dir_fin+'/'+model+'_train_loss_accuracy.txt','r') as f:
            f.readline()
            tr_loss = float(f.readline().strip()[6:])
        train_loss.append(tr_loss)

        with open(prel_dir_fin+'/'+model+'_validation_loss_accuracy.txt','r') as f:
            f.readline()
            val_loss = float(f.readline().strip()[6:])
        validation_loss.append(val_loss)

        with open(prel_dir_fin+'/'+model+'_test_ROC_auc.pkl','rb') as f:
            roc=pickle.load(f)
            rocs.append(roc)
            
    plt.figure(figsize=(14, 8))
    plt.grid(True)
    plt.plot(layers,train_loss,color='r',label='Train Loss')
    plt.plot(layers,validation_loss,color='b',label='Validation Loss',alpha=0.7)
    plt.plot(layers,test_loss,color='g',label='Test Loss',alpha=0.4)
    plt.xticks(layers)
    plt.xlabel('Layer Width')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.title('Train Test Validation Loss with depth '+str(depth)+' of PCA constant width models')
    plt.savefig('/home/private/Hepd/Dataset_4/analysis/PCA/constant/losses/depths/losses_PCA_depth_'+str(depth)+'.png')
    plt.close()

    plt.figure(figsize=(14, 8))
    plt.grid(True)
    plt.plot(layers,rocs,color='g',label='Test ROC area')
    plt.xticks(layers)
    plt.xlabel('Layer Width')
    plt.ylabel('ROC Value')
    plt.legend()
    plt.title('Test ROCs with depth '+str(depth)+' of PCA constant width models')
    plt.savefig('/home/private/Hepd/Dataset_4/analysis/PCA/constant/rocs/depths/rocs_PCA_depth_'+str(depth)+'.png')
    plt.close()


#### PLOTTIAMO LE ROC E LE LOSS PER DEPTH COSTANTE ####
prel_dir='/home/private/Hepd/Dataset_4/fcNN/PCA/constant/constant_PCA_depth_'
arch='constant'
rocs=[]
train_loss=[]
validation_loss=[]
for layer in layers:
    rocs=[]
    train_loss=[]
    validation_loss=[]
    test_loss=[]
    for depth in depths:
        model=arch+'_PCA_depth_'+str(depth)+'_'+str(layer)
        prel_dir_fin=prel_dir+str(depth)+'_'+str(layer)    
        with open(prel_dir_fin+'/'+model+'_test_loss_accuracy.txt','r') as f:
            f.readline()
            te_loss = float(f.readline().strip()[6:])
        test_loss.append(te_loss)

        with open(prel_dir_fin+'/'+model+'_train_loss_accuracy.txt','r') as f:
            f.readline()
            tr_loss = float(f.readline().strip()[6:])
        train_loss.append(tr_loss)

        with open(prel_dir_fin+'/'+model+'_validation_loss_accuracy.txt','r') as f:
            f.readline()
            val_loss = float(f.readline().strip()[6:])
        validation_loss.append(val_loss)

        with open(prel_dir_fin+'/'+model+'_test_ROC_auc.pkl','rb') as f:
            roc=pickle.load(f)
            rocs.append(roc)
    plt.figure(figsize=(14, 8))
    plt.grid(True)
    plt.plot(depths,train_loss,color='r',label='Train Loss')
    plt.plot(depths,validation_loss,color='b',label='Validation Loss',alpha=0.7)
    plt.plot(depths,test_loss,color='g',label='Test Loss',alpha=0.4)
    plt.xticks(depths)
    plt.xlabel('Depth')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.title('Train Test Validation Loss with layers of '+str(layer)+' neurons of PCA constant width models')
    plt.savefig('/home/private/Hepd/Dataset_4/analysis/PCA/constant/losses/layers/losses_PCA_layer_'+str(layer)+'.png')
    plt.close()

    plt.figure(figsize=(14, 8))
    plt.grid(True)
    plt.plot(depths,rocs,color='g',label='Test ROC area')
    plt.xticks(depths)
    plt.xlabel('Depth')
    plt.ylabel('ROC Value')
    plt.legend()
    plt.title('Test ROCs with layers of '+str(layer)+' neurons of PCA constant width models')
    plt.savefig('/home/private/Hepd/Dataset_4/analysis/PCA/constant/rocs/layers/rocs_PCA_layer_'+str(layer)+'.png')
    plt.close()