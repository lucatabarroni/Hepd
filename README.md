# Hepd
Classificazione di Elettroni vs Protoni sia con Neural Network che con BDT

Il dataset è composto da 32 (in realtà 31) numeri di ADC e da 9 numeri di Lyso crystals.

Abbiamo utilizzato sia delle BDT che delle Neural Network Fully-Connected (con e senza ResNet) e delle Convolutional.

DESCRIZIONE DEI MODELLI:

Descrizione delle FCNN utilizzate:

per le FCNN abbiamo considerato tre tipi di architetture diverse: con larghezza crescente (bottle), con larfgezza decrescente (funnel) e con larghezza costante (constant). Per ogni tipo di architettura abbiamo considerato 11 diverse profondità:

- depth 4
- depth 5
- depth 7
- depth 9
- depth 10
- depth 11
- depth 13
- depth 14
- depth 16
- depth 18
- depth 20


queste profondità comprendono un layer di input e un layer di output denso da un singolo neurone.
Inoltre per ognuna di queste profondità abbiamo considerato 11 larghezze di layer "caratteristico". Per i modelli bottle il layer caratteristico è l'ultimo, per il funnel è il primo, per il constant il layer caratteristico sono tutti avendo tutti la stessa larghezza. Le larghezze di layer caratteristico testate sono:

- layer 32
- layer 54
- layer 64
- layer 94
- layer 128
- layer 170
- layer 210
- layer 256
- layer 320
- layer 410
- layer 512

Per i modelli bottle il layer iniziale ha 16 neuroni, mentre per i modelli funnel il layer finale ha 16 neuroni.
N.B. per i modelli bottle e funnel non esistono i modelli con depth 20 e layer caratteristico 32. Questo perché con 20 layer, partendo da 16 per i modelli bottle o arrivando ai 16 neuroni per i modelli funnel, la variazione del numero di neuroni previsto per layer è inferiore ad 1 e quindi non è possibile creare quei due modelli. Il numero totale di modelli è quindi 120(bottle)+120(funnel)+121(constant) = 361

Per ogni modello FCNN abbiamo anche realizzato la versione ResNet aggiungendo le skip connections. I modelli ResNet sono stati creati partendo dalle architetture FCNN con le tre diverse forme: Bottle, Funne, Constant. Abbiamo utilizzato le seguenti regole:

- i ResNet blocks hanno dimensione tutte uguali di due blocchi l'uno ("Deep Residual Learning for Image Recognition" He et al.)
- le skip connections partono dall'output del secondo layer e finiscono nell'input del penultimo ( o terzultimo nel caso di profondità con numero dispari di layer)
- in caso di varazione della dimensione tra input ed output del ResNet block (per i modelli bottle e funnel) abbiamo inserito un blocco denso sul ramo della skip connection per proiettare l'input sulla dimensione dell'output. Questi blocchi densi non hanno nessuna attivazione (funzione lineare)

Per come vengono aggiunti i ResNet blocks, partendo dal secondo e arrivando al penultimo, questi non possono essere aggiunti alle architetture con depth pari a 4 e 5. Quinidi per le architetture bottle e funnel sono 9 diverse depth per 11 layer caratteristivi 9x11 = 99 modelli, meno quello con depth 20 e layer 32 quindi 98. Per i modelli Constant abbiamo le 9 profondità per 11 layer caratteristici 9x11 = 99 modelli. Il totale è 98+98+99 = 295 modelli ResNet. 

Quindi in totale abbiamo creato e trainato 361 FCNN + 295 FCNNResNet = 656 modelli Fully Connected.


Le Bdt sono state create partendo da due parametri fissi : learning_rate = 0.05 e boostinng_type = 'gbdt'. Il Gradient Boosting Decision Tree è il normale algoritmo di gradient boosting in cui gli elementi dell'ensemble sono trainat per correggere l'errore del precedente. Il valore del learning rate è la metà di quello di default. Glii altri parametri sono:

- max_depth degli estimatori
- n_estimators
- num_leaves numero di nodi finali degli estimatori

Questi tre parametri sono stati fatti variare per portare avanti uno studio sistematico delle prestazioni al variare dei parametri delle bdt:

- max_depth = [-1, 30, 300, 3000]
- n_estimators = [30, 300, 3000]
- num_leaves = [30, 300, 3000]

Il numero totale di bdt quindi è 4x3x3 = 36 modelli. 


 








































