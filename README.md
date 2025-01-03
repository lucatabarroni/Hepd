# Hepd
Classificazione di Elettroni vs Protoni sia con Neural Network che con BDT

Il dataset è composto da 32 (in realtà 31) numeri di ADC e da 9 numeri di Lyso crystals.

Abbiamo utilizzato sia delle BDT che delle Neural Network Fully-Connected e Convolutional con e senza ResNet.

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


La bdt è fatta con lightgbm con parametri:

- no maximum depth
- number of leaves 30
- number of estimators 2500
- learning rate 0.05









































