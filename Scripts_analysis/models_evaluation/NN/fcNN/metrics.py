import os
import matplotlib.pyplot as plt
import pickle


#### CREIAMO I PLOT DELLE LOSS AL VARIARE DI PROFONDITà E LARGHEZZA DI PRIMO E ULTIMO LAYER PER I BOTTLE MODELS ####

list_bottle_models=os.listdir('fcNN/bottle/')
arch='bottle'

#### CREIAMO UNA LISTA DI INTERI CON LE POSSIBILI LARGHEZZE DELL'ULTIMO/PRIMO LAYER ####
layers=[]
for model in list_bottle_models:
    layer='empty'
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
for model in list_bottle_models:
    depth='empty'
    for i,c in enumerate(model[13:]):
        if c!='_':
            continue
        else:
            depth=model[13:13+i]
            break
    if int(depth) not in depths:
        depths.append(int(depth))
depths.sort()
layers.sort()

#### PLOTTIAMO LE ROC E LE LOSS PER LAYER COSTANTE ####
#### DA QUESTA DIR ANDREMO A PRENDERE I VALORI DA PLOTTARE ####
prel_dir='fcNN/bottle/bottle_depth_'
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
        model=arch+'_depth_'+str(depth)+'_'+str(layer)
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
    plt.title('Train Test Validation Loss with depth '+str(depth)+' of bottle models')
    plt.savefig('analysis/bottle/losses/depths/losses_depth_'+str(depth)+'.png')
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
    plt.title('Test ROCs with depth '+str(depth)+' of bottle models')
    plt.savefig('analysis/bottle/rocs/depths/rocs_depth_'+str(depth)+'.png')
    plt.close()


#### PLOTTIAMO LE ROC E LE LOSS PER DEPTH COSTANTE ####
prel_dir='fcNN/bottle/bottle_depth_'
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
        model=arch+'_depth_'+str(depth)+'_'+str(layer)
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
    plt.title('Train Test Validation Loss with last layer of '+str(layer)+' neurons of bottle models')
    plt.savefig('analysis/bottle/losses/layers/losses_layer_'+str(layer)+'.png')
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
    plt.title('Test ROCs with last layer of '+str(layer)+' neurons of bottle models')
    plt.savefig('analysis/bottle/rocs/layers/rocs_layer_'+str(layer)+'.png')
    plt.close()



#### CREIAMO I PLOT DELLE LOSS AL VARIARE DI PROFONDITà E LARGHEZZA DI PRIMO E ULTIMO LAYER DEI FUNNEL LAYERS ####

list_funnel_models=os.listdir('fcNN/funnel/')
arch='funnel'

#### CREIAMO UNA LISTA DI INTERI CON LE POSSIBILI LARGHEZZE DELL'ULTIMO/PRIMO LAYER ####
layers=[]
for model in list_funnel_models:
    layer='empty'
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
for model in list_funnel_models:
    depth='empty'
    for i,c in enumerate(model[13:]):
        if c!='_':
            continue
        else:
            depth=model[13:13+i]
            break
    if int(depth) not in depths:
        depths.append(int(depth))
depths.sort()
layers.sort()

#### PLOTTIAMO LE ROC E LE LOSS PER LAYER COSTANTE ####
prel_dir='fcNN/funnel/funnel_depth_'
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
        model=arch+'_depth_'+str(depth)+'_'+str(layer)
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
    plt.title('Train Test Validation Loss with depth '+str(depth)+' of funnel models')
    plt.savefig('analysis/funnel/losses/depths/losses_depth_'+str(depth)+'.png')
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
    plt.title('Test ROCs with depth '+str(depth)+' of funnel models')
    plt.savefig('analysis/funnel/rocs/depths/rocs_depth_'+str(depth)+'.png')
    plt.close()


#### PLOTTIAMO LE ROC E LE LOSS PER DEPTH COSTANTE ####
prel_dir='fcNN/funnel/funnel_depth_'
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
        model=arch+'_depth_'+str(depth)+'_'+str(layer)
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
    plt.title('Train Test Validation Loss with first layer of '+str(layer)+' neurons of funnel models')
    plt.savefig('analysis/funnel/losses/layers/losses_layer_'+str(layer)+'.png')
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
    plt.title('Test ROCs with first layer of '+str(layer)+' neurons of funnel models')
    plt.savefig('analysis/funnel/rocs/layers/rocs_layer_'+str(layer)+'.png')
    plt.close()


#### CREIAMO I PLOT DELLE LOSS AL VARIARE DI PROFONDITà E LARGHEZZA DI PRIMO E ULTIMO LAYER DEI CONSTANT MODELS ####

list_constant_models=os.listdir('fcNN/constant/')
arch='constant'

#### CREIAMO UNA LISTA DI INTERI CON LE POSSIBILI LARGHEZZE DELL'ULTIMO/PRIMO LAYER ####
layers=[]
for model in list_constant_models:
    layer='empty'
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
for model in list_constant_models:
    depth='empty'
    for i,c in enumerate(model[15:]):
        if c!='_':
            continue
        else:
            depth=model[15:15+i]
            break
    if int(depth) not in depths:
        depths.append(int(depth))
depths.sort()
layers.sort()

#### PLOTTIAMO LE ROC E LE LOSS PER LAYER COSTANTE ####
prel_dir='fcNN/constant/constant_depth_'
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
        model=arch+'_depth_'+str(depth)+'_'+str(layer)
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
    plt.title('Train Test Validation Loss with depth '+str(depth)+' of constant width models')
    plt.savefig('analysis/constant/losses/depths/losses_depth_'+str(depth)+'.png')
    plt.close()

    plt.figure(figsize=(14, 8))
    plt.grid(True)
    plt.plot(layers,rocs,color='g',label='Test ROC area')
    plt.xticks(layers)
    plt.xlabel('Layer Width')
    plt.ylabel('ROC Value')
    plt.legend()
    plt.title('Test ROCs with depth '+str(depth)+' of constant width models')
    plt.savefig('analysis/constant/rocs/depths/rocs_depth_'+str(depth)+'.png')
    plt.close()


#### PLOTTIAMO LE ROC E LE LOSS PER DEPTH COSTANTE ####
prel_dir='fcNN/constant/constant_depth_'
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
        model=arch+'_depth_'+str(depth)+'_'+str(layer)
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
    plt.title('Train Test Validation Loss with layers of '+str(layer)+' neurons of constant width models')
    plt.savefig('analysis/constant/losses/layers/losses_layer_'+str(layer)+'.png')
    plt.close()

    plt.figure(figsize=(14, 8))
    plt.grid(True)
    plt.plot(depths,rocs,color='g',label='Test ROC area')
    plt.xticks(depths)
    plt.xlabel('Depth')
    plt.ylabel('ROC Value')
    plt.legend()
    plt.title('Test ROCs with layers of '+str(layer)+' neurons of constant width models')
    plt.savefig('analysis/constant/rocs/layers/rocs_layer_'+str(layer)+'.png')
    plt.close()