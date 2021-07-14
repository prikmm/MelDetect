import matplotlib.pyplot as plt
import numpy as np
import config

IMG_SIZES = config.IMG_SIZES
INC2018 = config.INC2018
INC2019 = config.INC2019
INC2020 = config.INC2020

def draw_plot(fold, history, model_name):          
    plt.figure(figsize=(15,5))
    plt.plot(np.arange(len(history.history['auc'])),history.history['auc'],'-o',label='Train AUC',color='#ff7f0e')
    plt.plot(np.arange(len(history.history['val_auc'])),history.history['val_auc'],'-o',label='Val AUC',color='#1f77b4')
    x = np.argmax( history.history['val_auc'] ); y = np.max( history.history['val_auc'] )
    xdist = plt.xlim()[1] - plt.xlim()[0]; ydist = plt.ylim()[1] - plt.ylim()[0]
    plt.scatter(x,y,s=200,color='#1f77b4'); plt.text(x-0.03*xdist,y-0.13*ydist,'max auc\n%.2f'%y,size=14)
    plt.ylabel('AUC',size=14); plt.xlabel('Epoch',size=14)
    plt.legend(loc=2)
    plt2 = plt.gca().twinx()
    plt2.plot(np.arange(len(history.history['loss'])),history.history['loss'],'-o',label='Train Loss',color='#2ca02c')
    plt2.plot(np.arange(len(history.history['val_loss'])),history.history['val_loss'],'-o',label='Val Loss',color='#d62728')
    x = np.argmin( history.history['val_loss'] ); y = np.min( history.history['val_loss'] )
    ydist = plt.ylim()[1] - plt.ylim()[0]
    plt.scatter(x,y,s=200,color='#d62728'); plt.text(x-0.03*xdist,y+0.05*ydist,'min loss',size=14)
    plt.ylabel('Loss',size=14)
    plt.title('FOLD %i - Image Size %i, %s, inc2018=%i, inc2019=%i, inc2020=%i'%
            (fold+1, IMG_SIZES[fold], model_name, INC2018[fold], INC2019[fold], INC2020[fold]), size=18)
    plt.legend(loc=3)
    plt.show()