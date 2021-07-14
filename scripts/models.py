import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
import tensorflow_addons as tfa 
import efficientnet.tfkeras as efn
import os
import config

AUTO = tf.data.experimental.AUTOTUNE

def build_mini_featvec_model(model_name="efficientnet_B0", dim=384, lr=None, image_only=True,
                             compile_model=True, metadata_length=9, feature_vec_size=1000,
                             normalize_features=True, return_feature_model=False,
                             inp_present=False, inp_layer=None, pretrained_model_weights=None,
                             effnet_pretrained_weights="imagenet", final_model=False):
    
    if not final_model:
        print("Starting to build the Mini-Featvec Model.........")
    
    model_dict = {
        "efficientnet_B0": efn.EfficientNetB0,
        "efficientnet_B1": efn.EfficientNetB1,
        "efficientnet_B2": efn.EfficientNetB2,
        "efficientnet_B3": efn.EfficientNetB3,
        "efficientnet_B4": efn.EfficientNetB4,
        "efficientnet_B5": efn.EfficientNetB5,
        "efficientnet_B6": efn.EfficientNetB6,
        "resnet50": tf.keras.applications.ResNet50,
        "vgg19": tf.keras.applications.VGG19,
        "Xception": tf.keras.applications.Xception,
    }   
    
    # For Image Input
    if inp_present:
        img_inp = inp_layer
    else:
        img_inp = keras.layers.Input(shape=(dim, dim, 3), name="img_input")
        
    base = model_dict[model_name](input_shape=(dim,dim,3),
                                  weights=effnet_pretrained_weights,
                                  include_top=False)(img_inp) 
    
    if not final_model:
        print(f"Created the {model_name} base......")

    pooling_layer = keras.layers.GlobalAveragePooling2D()(base)
    
    if not final_model:
        print("Created the pooling layer.......")

    if not image_only:
        # For metadata input
        metadata_inp = keras.layers.Input(shape=(metadata_length), name="metadata_input")
        dense_1 = keras.layers.Dense(512)(metadata_inp)
        dense_2 = keras.layers.Dense(256)(dense_1)
        dense_3 = keras.layers.Dense(64)(dense_2)
        # Concating the pooled features and metadata
        concat = keras.layers.Concatenate()([dense_3, pooling_layer])

        # A dense layer which will try to find a relation between image features and metadata
        feature_layer = keras.layers.Dense(feature_vec_size, activation="selu", name="featvec")(concat)

        # Normalizing the features
        normalized_feature = keras.layers.BatchNormalization(name="norm_featvec")(feature_layer)

        # Output
        output = keras.layers.Dense(1, activation="sigmoid", name="output")(normalized_feature)
    else:
        feature_layer = pooling_layer
        normalized_feature = keras.layers.BatchNormalization(name="norm_featvec")(pooling_layer)
        output = keras.layers.Dense(1,activation='sigmoid')(pooling_layer)
    
    if not final_model:
        print("Created all the layers.........")
    
    if normalize_features:
        feat_output = normalized_feature
    else:
        feat_output = feature_layer
            
    if image_only:
        if return_feature_model:
            featext_model = keras.Model(inputs=[img_inp], outputs=[feat_output])
        model = keras.Model(inputs=[img_inp], outputs=[output])
    else:
        if return_feature_model:
            featext_model = keras.Model(inputs=[metadata_inp, img_inp], outputs=[feat_output])
        model = keras.Model(inputs=[metadata_inp, img_inp], outputs=[output])
        
    if not final_model:
        print("Built the model..........")

    if pretrained_model_weights:
        model.load_weights(pretrained_model_weights)
        print("Loaded the pretrained weights...........")
        
    if compile_model:
        if lr:
            optimizer = keras.optimizers.Nadam(lr)
        else:
            optimizer = keras.optimizers.Nadam()
        model.compile(loss=tfa.losses.SigmoidFocalCrossEntropy(),
                      optimizer=keras.optimizers.Nadam(),
                      metrics=['AUC'])

    if return_feature_model:
        return model, featext_model, output, feat_output
    else:
        return model


def build_complete_ensemble_model(featvec_model_paths, inp_present=False, input_layer=None, dim=384,
                                  lr=None, normalize_features=True, final_model=True,
                                  compile_ensemble_model=True, concat_layer_name=None,
                                  featvec_layer_name=None):
    
    models = list()

    if not inp_present:
        input_layer = keras.layers.Input(shape=(dim, dim, 3), name="img_input")
    
    for model_path in featvec_model_paths:
        full_filename = os.path.basename(model_path)
        if model_path.find("feat") == -1:
            is_feat_path = False
            if len(full_filename) == 28:
                model_fold_name = f"{full_filename[0:15]}_{full_filename[-8]}_full"
                fold = model_fold_name[-6]
            else:
                model_fold_name = f"{full_filename[0:15]}_{full_filename[-9:-7]}_full"
                fold = model_fold_name[-7:-5]
            model_name = model_fold_name[:15]
            model_version = model_name[-2:]
        else:
            is_feat_path = True
            model_fold_name = full_filename[5:-3]
            model_name = model_fold_name[:-2]
            model_version = model_name[-2:]
            fold = model_fold_name[-1]
        
        full_model, featvec_model, full_model_output, featvec_output = build_mini_featvec_model(model_name=model_name, dim=dim,
                                                                                                feature_vec_size=2000, return_feature_model=True,
                                                                                                image_only=config.IMG_ONLY, inp_present=True,
                                                                                                inp_layer=input_layer,
                                                                                                normalize_features=normalize_features,
                                                                                                effnet_pretrained_weights=None,
                                                                                                final_model=final_model,
                                                                                                compile_model=False)
        
                  
        if is_feat_path:
            if compile_ensemble_model:
                featvec_model.load_weights(model_path)
            featvec_model.layers[1]._name = model_fold_name
            featvec_model.layers[2]._name = f"global_avg_pooling_2d_{model_version}_{fold}"
            featvec_model.layers[3]._name = f"{featvec_model.layers[3].name}_{model_version}_{fold}"
            featvec_model.trainable = False
            models.append(featvec_output)
        else:
            if compile_ensemble_model:
                full_model.load_weights(model_path)
            full_model.layers[1]._name = model_fold_name
            ##--
            full_model.layers[2]._name = f"global_avg_pooling_2d_{model_version}_{fold}"
            full_model.layers[3]._name = f"{full_model.layers[3].name}_{model_version}_{fold}"
            ##
            effnet_layer = full_model.layers[1](input_layer)
            global_avg_layer = full_model.layers[2](effnet_layer)
            ##--
            #featvec_output = keras.layers.BatchNormalization()(global_avg_layer)
            if normalize_features:
                normalize_layer = full_model.layers[3]
                featvec_output = normalize_layer(global_avg_layer)
            ##
            else:
                featvec_output = global_avg_layer
            """
            if normalize_features:
                featvec_output = keras.layers.BatchNormalization(
                    name=f"{featvec_model.layers[3].name}_{model_version}_{fold}")(global_avg_layer)
            """
            new_featvec_model = keras.Model(inputs=input_layer, outputs=featvec_output)
            if normalize_features:
                non_trainable_layers = new_featvec_model.layers[:-1]
            else:
                non_trainable_layers = new_featvec_model.layers

            for layer in non_trainable_layers:
                layer.trainable = False
            models.append(featvec_output)
        
        #keras.backend.clear_session()

    if concat_layer_name:
        concat_layer = concat_layer = keras.layers.Concatenate(name=concat_layer_name)(models)
    else:
        concat_layer = keras.layers.Concatenate()(models)

    if featvec_layer_name:
        dense_3 = keras.layers.Dense(1024, activation="selu",
                                     name=featvec_layer_name)(concat_layer)
    else:
        dense_3 = keras.layers.Dense(1024, activation="selu")(concat_layer)
    output_layer = keras.layers.Dense(1, activation="sigmoid")(dense_3)
    
    featvec_ensemble_model = keras.Model(inputs=[input_layer], outputs=[dense_3])
    ensemble_model = keras.Model(inputs=[input_layer], outputs=[output_layer])

    keras.utils.plot_model(ensemble_model)
    #if fine_tuning:
    #    ensemble_model.load_weights(ensemble_weights)


    if compile_ensemble_model:
        if lr:
            optimizer = keras.optimizers.Nadam(lr)
        else:
            optimizer = keras.optimizers.Nadam()
        
        ensemble_model.compile(loss=tfa.losses.SigmoidFocalCrossEntropy(),
                            optimizer=optimizer,
                            metrics=['AUC'])
        
    return ensemble_model, featvec_ensemble_model, output_layer, dense_3



def build_ensemble_of_ensemble(all_weights_list, dim=384, final_model=True,
                               lr=None, normalize_features=True):
    

    ensemble_models_list = []

    input_layer = keras.layers.Input(shape=(dim, dim, 3), name="img_input")

    for i, ensemble_n_mini_featvec_weights in enumerate(all_weights_list):
        ensemble_weights, mini_featvec_weights = ensemble_n_mini_featvec_weights
        inner_concat_layer_name = f"concat_layer_{i}"
        inner_featvec_layer_name = f"inner_featvec_layer_{i}"
        ensemble_model, ensemble_featvec, ensemble_output, featvec_ensemble_output = build_complete_ensemble_model(mini_featvec_weights,
                                                                                                                   dim=384, 
                                                                                                                   compile_ensemble_model=False,
                                                                                                                   inp_present=True,
                                                                                                                   input_layer=input_layer,
                                                                                                                   concat_layer_name=inner_concat_layer_name,
                                                                                                                   featvec_layer_name=inner_featvec_layer_name)
        ensemble_model.load_weights(ensemble_weights)
        for layer in ensemble_model.layers:
            layer.trainable = False
            
        ensemble_models_list.append(featvec_ensemble_output)
        
        
    concat_layer = keras.layers.Concatenate()(ensemble_models_list)
    featvec_layer = keras.layers.Dense(1024, activation="selu")(concat_layer)
    output_layer = keras.layers.Dense(1, activation="sigmoid")(featvec_layer)

    featvec_ens_ens_model = keras.Model(inputs=[input_layer], outputs=[featvec_layer])
    complete_ens_ens_model = keras.Model(inputs=[input_layer], outputs=[output_layer])

    if lr:
        optimizer = keras.optimizers.Nadam(lr)
    else:
        optimizer = keras.optimizers.Nadam()
    
    complete_ens_ens_model.compile(loss=tfa.losses.SigmoidFocalCrossEntropy(),
                                   optimizer=optimizer,
                                   metrics=['AUC'])
    
    keras.utils.plot_model(complete_ens_ens_model)
        
    return complete_ens_ens_model, featvec_ens_ens_model, output_layer, featvec_layer

# ---------------------------------------------------------BELOW FUNCTIONS ARE FOR IMAGE SEGMENTATION:---------------------------------------------------------


def conv2d_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True):
    """Function to add 2 convolutional layers with the parameters passed to it"""
    # first layer
    x = keras.layers.Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
                            kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    
    # second layer
    x = keras.layers.Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
                            kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    
    return x
  

def get_unet(input_img, n_filters = 16, dropout = 0.1, batchnorm = True, output_channels=3):
    # Contracting Path
    c1 = conv2d_block(input_img, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    p1 = keras.layers.MaxPooling2D((2, 2))(c1)
    p1 = keras.layers.Dropout(dropout)(p1)

    c2 = conv2d_block(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    p2 = keras.layers.MaxPooling2D((2, 2))(c2)
    p2 = keras.layers.Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    p3 = keras.layers.MaxPooling2D((2, 2))(c3)
    p3 = keras.layers.Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    p4 = keras.layers.MaxPooling2D((2, 2))(c4)
    p4 = keras.layers.Dropout(dropout)(p4)

    c5 = conv2d_block(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm)

    # Expansive Path
    u6 = keras.layers.Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c5)
    u6 = keras.layers.Concatenate()([u6, c4])
    u6 = keras.layers.Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)

    u7 = keras.layers.Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c6)
    u7 = keras.layers.Concatenate()([u7, c3])
    u7 = keras.layers.Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)

    u8 = keras.layers.Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c7)
    u8 = keras.layers.Concatenate()([u8, c2])
    u8 = keras.layers.Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)

    u9 = keras.layers.Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)
    u9 = keras.layers.Concatenate()([u9, c1])
    u9 = keras.layers.Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)

    seg_output = keras.layers.Conv2D(output_channels, (1, 1), activation='sigmoid', name="seg_output")(c9)
    model = keras.Model(inputs=[input_img], outputs=[seg_output])
    return model, seg_output