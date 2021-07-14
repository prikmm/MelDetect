# USE VERBOSE=0 for silent, VERBOSE=1 for interactive, VERBOSE=2 for commit
import tensorflow as tf
from tensorflow import keras
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras.backend as K
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import models
import datasets
import images
import config
import callbacks
import plot
import os
import gc
import tensorflow_addons as tfa

IMG_ONLY = config.IMG_ONLY
strategy = config.strategy,
EPOCHS = config.EPOCHS,
FOLDS = config.FOLDS,
IMG_SIZES = config.IMG_SIZES,
BATCH_SIZES = config.BATCH_SIZES,
REPLICAS = config.REPLICAS,
SEED = config.SEED,
DATA_PATH = config.DATA_PATH,
DATA_PATH2 = config.DATA_PATH2,
DATA_PATH3 = config.DATA_PATH3,
TTA = config.TTA,
VERBOSE = config.VERBOSE,
DISPLAY_PLOT = config.DISPLAY_PLOT,
GRID_MASK = config.GRID_MASK,
GRID_MASK_AUG = config.GRID_MASK_AUG,



############################
# USE VERBOSE=0 for silent, VERBOSE=1 for interactive, VERBOSE=2 for commit
def train_models(ens_ens_model=False, ensemble_model=False, featvec_model_paths=None, model_name=None,
                 EPOCHS=EPOCHS, FOLDS=FOLDS, IMG_SIZES=IMG_SIZES, BATCH_SIZES=BATCH_SIZES, REPLICAS=REPLICAS,
                 SEED=SEED, DATA_PATH=DATA_PATH, DATA_PATH2=DATA_PATH2, DATA_PATH3=DATA_PATH3, TTA=TTA,
                 VERBOSE=VERBOSE, DISPLAY_PLOT=DISPLAY_PLOT, lr=None, add_lr_callback=True,
                 add_early_stopping_callback=True, lr_min=None, lr_max=None, lr_start=0.000005,
                 fine_tuning=False, ensemble_weights=None, grid_mask=True, grid_mask_aug=True,
                 test_model=False, pretrained_mini_featvec_model_weights=None):
                 #files_train=files_train, files_test=files_test):
    
    skf = KFold(n_splits=FOLDS,shuffle=True,random_state=SEED)
    oof_pred = []; oof_tar = []; oof_val = []; oof_names = []; oof_folds = [] 
    best_auc = 0
    best_fold = None
    best_model_history = None
    #preds = np.zeros((count_data_items(files_test),1))

    for fold,(idxT,idxV) in enumerate(skf.split(np.arange(15))):

        if ens_ens_model or ensemble_model:
            if fold > 0:
                continue

        # DISPLAY FOLD INFO            
        files_train, files_valid, files_test = datasets.get_file_paths(model_name, fold, idxT, idxV)

        #if final_model:
        #    print("Train Files:", files_train)
        #    print("Valid Files:", files_valid)
        #    print("Test Files:", files_test)

        # BUILD MODEL
        K.clear_session()

        # Getting DATASETS:
        train_dataset = datasets.get_dataset(files_train, augment=True, shuffle=True, repeat=True,
                                    dim=IMG_SIZES[fold],batch_size = BATCH_SIZES[fold],
                                    image_only=IMG_ONLY, grid_mask=grid_mask,
                                    grid_mask_aug=grid_mask_aug)


        valid_dataset = datasets.get_dataset(files_valid, augment=True, shuffle=True,
                                    repeat=False,dim=IMG_SIZES[fold],
                                    image_only=IMG_ONLY, grid_mask=grid_mask,
                                    grid_mask_aug=grid_mask_aug)

        
        with strategy.scope():
            if ens_ens_model:
                model, feat_model, _, _ = models.build_ensemble_of_ensemble(all_weights_list=featvec_model_paths,
                                                                     dim=IMG_SIZES[fold],
                                                                     final_model=True, lr=lr,
                                                                     normalize_features=True)
            elif ensemble_model:
                model, feat_model,_, _ = models.build_complete_ensemble_model(featvec_model_paths, dim=IMG_SIZES[fold], lr=lr)
            else:
                model, feat_model, _, _ = models.build_mini_featvec_model(model_name=model_name, dim=IMG_SIZES[fold],
                                                                   feature_vec_size=2000, return_feature_model=True,
                                                                   image_only=IMG_ONLY, lr=lr,
                                                                   pretrained_model_weights=pretrained_mini_featvec_model_weights,
                                                                   final_model=False)


        if fine_tuning:
            with strategy.scope():
                print("Loading Ensemble Weights for Fine-Tuning........")
                model.load_weights(ensemble_weights)
                print("Ensemble Weights loaded, now, making layers \"Trainable\"...........") 
                for layer in model.layers:
                    if not layer.trainable:
                        layer.trainable = True
                if lr:
                    optimizer = keras.optimizers.Nadam(lr)
                else:
                    optimizer = keras.optimizers.Nadam()
                print("Compiling the new model............")
                model.compile(loss=tfa.losses.SigmoidFocalCrossEntropy(),
                              optimizer=optimizer,
                              metrics=['AUC'])
                
            checkpoint_file_path = callbacks.get_model_checkpoint_path(f'FT_{model_name}', fold, IMG_SIZES[fold])
            sv = tf.keras.callbacks.ModelCheckpoint(
                f'{checkpoint_file_path}.h5', monitor='val_loss', verbose=0, save_best_only=True,
                save_weights_only=True, mode='min', save_freq='epoch')

            callbacks = [sv]
            if add_lr_callback:
                callbacks += [callbacks.get_lr_callback(BATCH_SIZES[fold],
                                             lr_start=lr_start,
                                             lr_min=lr_min,
                                             lr_max=lr_max)]
            
            if add_early_stopping_callback:
                callbacks += [keras.callbacks.EarlyStopping(patience=5)]

            callbacks += [keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                                            patience=3, min_lr=lr_min)]
            print('Fine Tuning...')
            history = model.fit(
                    train_dataset,
                    epochs=EPOCHS[fold],
                    validation_data=valid_dataset, 
                    callbacks = callbacks, 
                    steps_per_epoch=datasets.count_data_items(files_train)/BATCH_SIZES[fold]//REPLICAS,
                    verbose=VERBOSE,
            )
        else:
            checkpoint_file_path = callbacks.get_model_checkpoint_path(model_name, fold, IMG_SIZES[fold])
            sv = tf.keras.callbacks.ModelCheckpoint(
                f'{checkpoint_file_path}.h5', monitor='val_loss', verbose=0, save_best_only=True,
                save_weights_only=True, mode='min', save_freq='epoch')
            
            callbacks = [sv]
            if add_lr_callback:
                callbacks += [callbacks.get_lr_callback(BATCH_SIZES[fold],
                                              lr_start=lr_start,
                                              lr_min=lr_min,
                                              lr_max=lr_max)]
            
            if add_early_stopping_callback:
                callbacks += [keras.callbacks.EarlyStopping(patience=5)]

            #callbacks += [keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
            #                                                patience=3, min_lr=lr_min)]

            print('Training...')
            history = model.fit(
                    train_dataset,
                    epochs=EPOCHS[fold],
                    validation_data=valid_dataset, 
                    callbacks = callbacks, 
                    steps_per_epoch=datasets.count_data_items(files_train)/BATCH_SIZES[fold]//REPLICAS,
                    verbose=VERBOSE,
            )
        
        if not (ens_ens_model or ensemble_model):
            print('Loading best model...')
            model.load_weights(f'{checkpoint_file_path}.h5')
            head_tail = os.path.split(f'{checkpoint_file_path}.h5')
            feat_model.save_weights(f'{os.path.join(head_tail[0], f"feat_{model_name}_{fold}")}.h5')
            np.save(os.path.join(f'{checkpoint_file_path}_history.pkl'), history.history)
        
        if test_model:

            # PREDICT OOF USING TTA
            print('Predicting OOF with TTA...')
            valid_dataset = datasets.get_dataset(files_valid,labeled=False,return_image_names=False,augment=True,
                                        repeat=True,shuffle=False,dim=IMG_SIZES[fold],batch_size=BATCH_SIZES[fold]*4,
                                        image_only=IMG_ONLY, grid_mask=grid_mask,
                                        grid_mask_aug=grid_mask_aug)
            ct_valid = datasets.count_data_items(files_valid)
            STEPS = TTA * ct_valid/BATCH_SIZES[fold]/4/REPLICAS
            pred = model.predict(valid_dataset,steps=STEPS,verbose=VERBOSE)[:TTA*ct_valid,] 
            oof_pred.append(np.mean(pred.reshape((ct_valid,TTA),order='F'),axis=1) )                 
            #oof_pred.append(model.predict(get_dataset(files_valid,dim=IMG_SIZES[fold]),verbose=1))

            # GET OOF TARGETS AND NAMES
            valid_dataset = datasets.get_dataset(files_valid, augment=False, repeat=False, dim=IMG_SIZES[fold],
                                        labeled=True, return_image_names=True, image_only=IMG_ONLY, grid_mask=grid_mask,
                                        grid_mask_aug=grid_mask_aug)
            oof_tar.append(np.array([target.numpy() for img, target in iter(valid_dataset.unbatch())]))
            oof_folds.append(np.ones_like(oof_tar[-1],dtype='int8')*fold)
            valid_dataset = datasets.get_dataset(files_valid, augment=False, repeat=False, dim=IMG_SIZES[fold],
                                        labeled=False, return_image_names=True, image_only=IMG_ONLY)
            oof_names.append(np.array([feature_dict[1].numpy().decode("utf-8") for feature_dict in iter(valid_dataset.unbatch())]))

            # PREDICT TEST USING TTA
            print('Predicting Test with TTA...')
            test_dataset = datasets.get_dataset(files_test,labeled=False,return_image_names=False,augment=True,
                                    repeat=True,shuffle=False,dim=IMG_SIZES[fold],batch_size=BATCH_SIZES[fold]*4,
                                    image_only=IMG_ONLY, grid_mask=grid_mask,
                                        grid_mask_aug=grid_mask_aug)
            ct_test = datasets.count_data_items(files_test)
            STEPS = TTA * ct_test/BATCH_SIZES[fold]/4/REPLICAS
            pred = model.predict(test_dataset,steps=STEPS,verbose=VERBOSE)[:TTA*ct_test,] 
            #preds[:,0] += np.mean(pred.reshape((ct_test,TTA),order='F'),axis=1) * WGTS[fold]

            # REPORT RESULTS
            auc = roc_auc_score(oof_tar[-1],oof_pred[-1])
            if auc > best_auc:
                best_fold = fold
                best_model_history = history
            oof_val.append(np.max( history.history['val_auc'] ))
            print('#### FOLD %i OOF AUC without TTA = %.3f, with TTA = %.3f'%(fold+1,oof_val[-1],auc))

            # PLOT TRAINING
            if DISPLAY_PLOT:
                plot.draw_plot(fold, history, model_name)

        del model
        keras.backend.clear_session()
        gc.collect()
    
    #root_checkpoint_dir = "my_checkpoints"
    with strategy.scope():
        if ens_ens_model or ensemble_model:
            return best_model_history
        else:
            model, feat_model, _, _ = models.build_mini_featvec_model(model_name=model_name, dim=IMG_SIZES[best_fold],
                                                                feature_vec_size=2000, return_feature_model=True,
                                                                image_only=IMG_ONLY, lr=lr)
            #model = build_model(model_name=model_name, dim=IMG_SIZES[best_fold])
            model.load_weights(os.path.join(callbacks.root_checkpoint_dir, model_name,
                                            f'{model_name}_fold{best_fold}_{IMG_SIZES[best_fold]}.h5'))

            return model, best_model_history

#####################

