import tensorflow as tf
from tensorflow.python.lib.io import file_io
import numpy as np
from functools import partial
import images
import os
import cv2
import re
import config
import matplotlib.pyplot as plt

AUTO = tf.data.experimental.AUTOTUNE

IMG_SIZES = config.IMG_SIZES
BATCH_SIZES = config.BATCH_SIZES
DATA_PATH = config.DATA_PATH
DATA_PATH2 = config.DATA_PATH2
DATA_PATH3 = config.DATA_PATH3
IMG_ONLY = config.IMG_ONLY
REPLICAS = config.REPLICAS
ROOT_FEATVEC_PATH = config.ROOT_FEATVEC_PATH
SEG_TFREC_PATHS = config.SEG_TFREC_PATHS
INC2019 = config.INC2019
INC2018 = config.INC2018
INC2020 = config.INC2020
GRID_MASK = config.GRID_MASK
GRID_MASK_AUG = config.GRID_MASK_AUG


def get_file_paths(model_name, fold, idxT, idxV):
    print('#'*25); print('#### FOLD',fold+1)
    print('#### Image Size %i with %s and batch_size %i'%
          (IMG_SIZES[fold], model_name,BATCH_SIZES[fold]*REPLICAS))

    # CREATE TRAIN AND VALIDATION SUBSETS
    files_train = tf.io.gfile.glob([DATA_PATH[fold] + '/train%.2i*.tfrec'%x for x in idxT])
    if INC2019[fold]:
        files_train += tf.io.gfile.glob([DATA_PATH2[fold] + '/train%.2i*.tfrec'%x for x in idxT*2+1])
        print('#### Using 2019 external data')
    if INC2018[fold]:
        files_train += tf.io.gfile.glob([DATA_PATH2[fold] + '/train%.2i*.tfrec'%x for x in idxT*2])
        print('#### Using 2018+2017 external data')
    if INC2020[fold]:
        files_train += tf.io.gfile.glob([DATA_PATH3[fold] + '/train%.2i*.tfrec'%x for x in idxT + 15])
        print('#### Using 2020 external data')
    np.random.shuffle(files_train); print('#'*25)
    files_valid = tf.io.gfile.glob([DATA_PATH[fold] + '/train%.2i*.tfrec'%x for x in idxV])
    files_test = np.sort(np.array(tf.io.gfile.glob(DATA_PATH[fold] + '/test*.tfrec')))
    return files_train, files_valid, files_test


if GRID_MASK_AUG:
    root_weight_dir = os.path.join(ROOT_FEATVEC_PATH, "gm_aug_Featvec_models", "grid_mask_Featvec_models")
elif GRID_MASK:
    root_weight_dir = os.path.join(ROOT_FEATVEC_PATH, "no_gm_aug_Featvec_models", "no_gm_aug_Featvec_models")
else:
    root_weight_dir = os.path.join(ROOT_FEATVEC_PATH, "Featvec_models", "Featvec_models")

def get_model_weights(model_names, root_weight_dir=root_weight_dir, path_type="Full"):

    def get_featvec_paths(featvec_paths):
        for i, featvec_path in enumerate(featvec_paths):
            weight_file = file_io.FileIO(featvec_path, mode="rb")
            file_name = os.path.basename(featvec_path)
            if os.path.exists(file_name):
                print(f"{file_name} already exists..,")
            else:
                with open(file_name, "wb") as temp_model_file:
                    temp_model_file.write(weight_file.read())
            featvec_paths[i] = file_name
        return featvec_paths

    def get_full_model_paths(full_model_paths):
        for i, full_model_path in enumerate(full_model_paths):
            file_name = os.path.basename(full_model_path)
            weight_file = file_io.FileIO(full_model_path, mode="rb")
            if os.path.exists(file_name):
                print(f"{file_name} already exists..,")
            else:
                with open(file_name, "wb") as temp_model_file:
                    temp_model_file.write(weight_file.read())
            full_model_paths[i] = file_name
        return full_model_paths

    all_model_weight_paths = []
    for model_name in model_names:
        featvec_paths = tf.io.gfile.glob(os.path.join(root_weight_dir, "featvec_models",
                                                    model_name, "*.h5"))
        full_model_paths = tf.io.gfile.glob(os.path.join(root_weight_dir, "Full_models",
                                                            model_name, "*.h5"))
        if path_type == "All":
            all_model_weights_paths += get_featvec_paths(featvec_paths)
            all_model_weights_paths += get_full_model_paths(full_model_paths)
        elif path_type == "Feat":
            featvec_paths = get_featvec_paths(featvec_paths)
            all_model_weight_paths += featvec_paths
        elif path_type == "Full":
            full_model_paths = get_full_model_paths(full_model_paths)
            all_model_weight_paths += full_model_paths
        else:
            print(f"No Model Weights present for architecture - {model_name} and path_type - {path_type}")
    return all_model_weight_paths


def read_feat_tfrecord(example, full_model=False, image_only=IMG_ONLY,
                       test_set=False, labeled=True, return_image_names=False):
    if not test_set:
        if labeled:
            tfrec_format = {
                'image'                        : tf.io.FixedLenFeature([], tf.string),
                'image_name'                   : tf.io.FixedLenFeature([], tf.string),
                'patient_id'                   : tf.io.FixedLenFeature([], tf.int64),
                'sex'                          : tf.io.FixedLenFeature([], tf.int64),
                'age_approx'                   : tf.io.FixedLenFeature([], tf.int64),
                'anatom_site_general_challenge': tf.io.FixedLenFeature([], tf.int64),
                'target'                       : tf.io.FixedLenFeature([], tf.int64)
            }      
        else:
            tfrec_format = {
                'image'                        : tf.io.FixedLenFeature([], tf.string),
                'image_name'                   : tf.io.FixedLenFeature([], tf.string),
                'patient_id'                   : tf.io.FixedLenFeature([], tf.int64),
                'sex'                          : tf.io.FixedLenFeature([], tf.int64),
                'age_approx'                   : tf.io.FixedLenFeature([], tf.int64),
                'anatom_site_general_challenge': tf.io.FixedLenFeature([], tf.int64),
            }
    else:
        tfrec_format = {
                'image'                        : tf.io.FixedLenFeature([], tf.string),
                'image_name'                   : tf.io.FixedLenFeature([], tf.string),
                'patient_id'                   : tf.io.FixedLenFeature([], tf.int64),
                'sex'                          : tf.io.FixedLenFeature([], tf.int64),
                'age_approx'                   : tf.io.FixedLenFeature([], tf.int64),
                'anatom_site_general_challenge': tf.io.FixedLenFeature([], tf.int64),
        }
        
    example = tf.io.parse_single_example(example, tfrec_format)
    if full_model:
        if image_only:
            if labeled:
                return ({"seg_input": example['image'],
                         "img_input": example['image']},
                        {"img_names":example['image_name'],
                         "target": example['target']})
            else:
                return ({"seg_input": example['image'],
                         "img_input": example['image']},
                        {"img_names":example['image_name']})
        else:
            if not test_set:
                if labeled:
                    return ({"seg_input": example['image'],
                             "img_input": example['image'],
                             "metadata_input": [example['sex'], example['age_approx'],
                                                example['anatom_site_general_challenge']]},
                            {"img_names":example['image_name'],
                             "target": example['target']})
                
            return ({"seg_input": example['image'],
                     "img_input": example['image'],
                     "metadata_input": [example['sex'], example['age_approx'],
                                        example['anatom_site_general_challenge']]},
                     {"img_names":example['image_name']})
    else:
        if image_only:
            if labeled:
                return ({"img_input": example['image']}, example['target'])
            else:
                return ({"img_input": example['image']}, example['image_name'] if return_image_names else 0)
        else:
            if not test_set:
                if labeled:
                    return ({"img_input": example['image'],
                            "metadata_input": [example['sex'], example['age_approx'],
                                            example['anatom_site_general_challenge']]},
                            example['target'])
                
            return ({"img_input": example['image'],
                    "metadata_input": [example['sex'], example['age_approx'],
                                        example['anatom_site_general_challenge']]},
                    example['image_name'] if return_image_names else 0)
    


def metadata_handler(metadata):
    new_metadata = tf.one_hot(metadata[0], 2,
                              on_value=1.0, off_value=0.0,
                              axis=-1)
    new_metadata = tf.concat([new_metadata, tf.expand_dims(tf.cast(metadata[1], tf.float32), 0)], axis=0)
    new_metadata = tf.concat([new_metadata,
                              tf.one_hot(metadata[2], 6,
                                         on_value=1.0, off_value=0.0,
                                         axis=-1)],
                             axis=0)
    return new_metadata


def prepare_input_data(input_data, image_only=IMG_ONLY, dim=384, full_model=False):
    input_data["img_input"] = images.prepare_image(input_data["img_input"], dim=dim)
    if not image_only:
        input_data["metadata_input"] = metadata_handler(input_data["metadata_input"])
    if full_model:
        input_data["seg_input"] = images.prepare_image(input_data["seg_input"], dim=dim)
    return input_data



def get_dataset(files, image_only=IMG_ONLY, full_model=False, augment = False, grid_mask=True, 
                grid_mask_aug=False, shuffle = False, repeat = False, labeled=True, return_image_names=False,
                batch_size=16, dim=256, REPLICAS=REPLICAS):
    
    read_labeled_unlabeled = partial(read_feat_tfrecord, image_only=image_only,
                                     labeled=labeled, return_image_names=return_image_names,
                                     full_model=full_model)
    
    # In full model, we need the filenames
    # in the same order as the rest of the input.
    # Using, interleave does not preserve the order
    # between during the latter half of the processing(when we create an
                                                    #  image_names_dataset)
    if full_model:  
        ds = tf.data.TFRecordDataset(files)
        ds = ds.map(read_labeled_unlabeled, num_parallel_calls=AUTO)
    else:
        files_ds = tf.data.Dataset.list_files(files)
        ds = files_ds.interleave(lambda tfrec_path:
                                    tf.data.TFRecordDataset(tfrec_path).map(
                                        read_labeled_unlabeled, num_parallel_calls=AUTO),
                                num_parallel_calls=AUTO)


    if repeat and not full_model:
        ds = ds.repeat()
    
    if shuffle and not full_model: 
        ds = ds.shuffle(1024*8)
        opt = tf.data.Options()
        opt.experimental_deterministic = False
        ds = ds.with_options(opt)
    
    if not full_model:
        ds = ds.map(lambda input_data, target_or_image_names: (prepare_input_data(input_data,
                                                                                  image_only=image_only,
                                                                                  dim=dim),
                                                               target_or_image_names),
                    num_parallel_calls=AUTO)
                    
        if augment:
            augmenter_func = partial(images.featvec_augmenter, grid_mask=grid_mask, grid_mask_aug=grid_mask_aug)
            ds = ds.flat_map(augmenter_func)

    else:
        ds = ds.map(lambda input_data, target_n_image_names:(prepare_input_data(input_data,
                                                                                 image_only=image_only,
                                                                                 dim=dim,
                                                                                 full_model=full_model),
                                                             target_n_image_names),
                    num_parallel_calls=AUTO)   
    
    def target_handler(target_n_name):
        if isinstance(target_n_name, dict):
            target_n_name["target"] = tf.cast(target_n_name["target"], tf.float32)
            return target_n_name
        else:
            target_n_name = tf.cast(target_n_name, tf.float32)
            return target_n_name

    if labeled:
        ds = ds.map(lambda features, target_n_name: (features,
                                                     target_handler(target_n_name)),
                    num_parallel_calls=AUTO)

    if full_model:
    
    #------------THE 2 PLOTTING BLOCKS ARE TO CHECK WHETHER--------------------#
    #----------------THE INPUT ORDER IS PRESERVED OR NOT-----------------------# 
    # BLOCK-1:     -------------------------------------------------------------
        fig = plt.figure(figsize=(10, 7))
        rows = 2
        columns = 5
        for count, item in enumerate(ds.take(5)):
            x, y = item
            fig.add_subplot(rows, columns, count+1)
            print(f'BEFORE_{count+1}:', y["img_names"])
            plt.imshow(x["img_input"])
            plt.axis('off')
            plt.title(f'BEFORE_{count+1}:')
    #-------------------------------BLOCK-1 FINISH-----------------------------#

        image_names_ds = ds.map(lambda features, target_n_names: target_n_names["img_names"],
                                num_parallel_calls=AUTO)
        image_names_ds = image_names_ds.batch(batch_size * REPLICAS, drop_remainder=True)
        image_names_ds = image_names_ds.prefetch(AUTO)

        if labeled:
            ds = ds.map(lambda features, target_n_names:(features, target_n_names["target"]),
                        num_parallel_calls=AUTO)
        else:
            ds = ds.map(lambda features, target_n_names: features,
                        num_parallel_calls=AUTO)
        
        ds = ds.batch(batch_size * REPLICAS, drop_remainder=True)
        ds = ds.prefetch(AUTO)

    # BLOCK-2:     -------------------------------------------------------------
        for item in image_names_ds.take(1):
            for count, single_item in enumerate(item[:5]): 
                print(f'AFTER_{count+1}:', single_item)

        for item in ds.take(1):
            if labeled:
                images_set = item[0]["img_input"]
            else:
                images_set = item["img_input"]
            for count, single_item in enumerate(images_set[:5]):
                fig.add_subplot(rows, columns, columns+(count+1))
                plt.imshow(single_item)
                plt.axis('off')
                plt.title(f'AFTER_{count+1}:')

        plt.tight_layout(h_pad=-10)
    #-------------------------------BLOCK-2 FINISH-----------------------------#  

        return ds, image_names_ds
    else:
        return ds


def count_data_items(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) 
         for filename in filenames]
    return np.sum(n)


# --------------------BELOW FUNCTIONS ARE FOR IMAGE SEGMENTATION:---------------#
def get_seg_paths(data_type="train", tfrec_roots=SEG_TFREC_PATHS, img_root_paths=None):
    if data_type == "tfrecords":
        test_paths = []
        train_paths = []
        for tfrec_root in tfrec_roots:  
            test_paths += tf.io.gfile.glob(tfrec_root+'/test*.tfrec')
            train_paths += tf.io.gfile.glob(tfrec_root+'/train*.tfrec')
        test_paths = np.sort(np.array(test_paths))
        train_paths = np.sort(np.array(train_paths))
        return train_paths, test_paths
    else:
        complete_img_paths = [0]*len(img_root_paths)
        for index, img_root_path in enumerate(img_root_paths):
            complete_img_paths[index] = np.sort(np.array(tf.io.gfile.glob(img_root_path+ '/*.jpg')))
        return complete_img_paths

#All the code below comes from TensorFlow's docs here


def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def serialize_example(feature0, feature1, feature2):
  feature = {
      'image': _bytes_feature(feature0),
      'image_name': _bytes_feature(feature1),
      'mask': _bytes_feature(feature2),
  }
  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()


def create_seg_tfrecords(tfrecord_type="train", SIZE=500, img_root_paths=None):
    image_paths, mask_paths = get_seg_paths(data_type=tfrecord_type,
                                            img_root_paths=img_root_paths)
    folder_name = f'{tfrecord_type}_tfrecords'
    path = os.path.join(os.getcwd(), folder_name)
    os.makedirs(path, exist_ok=True)
    path_zip = zip(image_paths, mask_paths)
    tfrecord_nums = image_paths.size // 500 + 1
    for tfrecord_counter in range(tfrecord_nums):
        tfrecord_size = min(SIZE, image_paths.size-tfrecord_counter*SIZE)
        print('\nCreating {tfrecord_type}_{tfrecord_counter}.tfrec......'.format(
                            tfrecord_type=tfrecord_type,tfrecord_counter=tfrecord_counter))
        with tf.io.TFRecordWriter(os.path.join(path, f'{tfrecord_type}{tfrecord_counter}.tfrec')) as writer:
            for k in range(tfrecord_size):
                # processing image
                image = cv2.imread(image_paths[tfrecord_size * tfrecord_counter + k])
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                image = cv2.imencode('.jpg', image, (cv2.IMWRITE_JPEG_QUALITY, 94))[1].tobytes()

                # processing mask
                mask = cv2.imread(mask_paths[tfrecord_size * tfrecord_counter + k])
                mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
                mask = cv2.imencode('.jpg', mask, (cv2.IMWRITE_JPEG_QUALITY, 94))[1].tobytes()

                # extracting image name
                image_name = os.path.split(image_paths[tfrecord_size*tfrecord_counter+k])[1]
                image_name = image_name.split('.')[0]
                
                # writing the example
                example = serialize_example(image, str.encode(image_name), mask)
                writer.write(example)
                if k%100==0: print(k,', ',end='')



def read_seg_tfrecord(example, input_layer_name,
                      labeled, return_image_names=False):
    if labeled:
        tfrec_format = {
            'image'                        : tf.io.FixedLenFeature([], tf.string),
            'image_name'                   : tf.io.FixedLenFeature([], tf.string),
            'mask'                         : tf.io.FixedLenFeature([], tf.string),
        }      
    else:
        tfrec_format = {
            'image'                        : tf.io.FixedLenFeature([], tf.string),
            'image_name'                   : tf.io.FixedLenFeature([], tf.string),
        }
        
    example = tf.io.parse_single_example(example, tfrec_format)
    if labeled:
        return ({input_layer_name: example['image']}, example['mask'])
    else:
        return ({input_layer_name: example['image']},
                example['image_name'] if return_image_names else 0)



"""
def get_seg_dataset(files, full_model=True, shuffle = False, repeat = False,
                    labeled=True, input_layer_name="seg_input", return_image_names=False,
                    batch_size=32, dim=384, mask_channels=3):
    
    read_labeled_unlabeled = partial(read_seg_tfrecord,
                                     input_layer_name=input_layer_name,
                                     labeled=labeled)

    if full_model:  
        ds = tf.data.TFRecordDataset(files)
        ds = ds.cache()
        ds = ds.map(read_labeled_unlabeled, num_parallel_calls=AUTO)
    else:
        files_ds = tf.data.Dataset.list_files(files)
        ds = files_ds.interleave(lambda tfrec_path:
                                    tf.data.TFRecordDataset(tfrec_path).map(
                                        read_labeled_unlabeled, num_parallel_calls=AUTO),
                                num_parallel_calls=tf.data.AUTOTUNE)
    
    if repeat:
        ds = ds.repeat()
    
    if shuffle: 
        ds = ds.shuffle(1024*8)
        opt = tf.data.Options()
        opt.experimental_deterministic = False
        ds = ds.with_options(opt)
    
    ds = ds.map(read_labeled_unlabeled, num_parallel_calls=AUTO)
    ds = ds.map(lambda image, mask_or_image_name: (prepare_image(image, dim=dim),
                                                   mask_or_image_name),
                num_parallel_calls=AUTO)
    # for image dataset
    if labeled: 
        ds = ds.map(lambda image, mask: (image, prepare_image(mask, dim=dim,
                                                              mask=True,
                                                              mask_channels=mask_channels)),
                    num_parallel_calls=AUTO)
     
    ds = ds.batch(batch_size * REPLICAS, drop_remainder=True)
    ds = ds.prefetch(AUTO)
    return ds
"""


def get_seg_dataset(files, full_model=False, input_layer_name=None,
                    shuffle = False, repeat = False,
                    labeled=True, augment=True, return_image_names=False,
                    batch_size=32, dim=384, mask_channels=3):

    read_labeled_unlabeled = partial(read_seg_tfrecord,
                                     input_layer_name=input_layer_name,
                                     labeled=labeled)

    if full_model:  
        ds = tf.data.TFRecordDataset(files)
        ds = ds.cache()
        ds = ds.map(read_labeled_unlabeled, num_parallel_calls=AUTO)
    else:
        files_ds = tf.data.Dataset.list_files(files)
        ds = files_ds.interleave(lambda tfrec_path:
                                    tf.data.TFRecordDataset(tfrec_path).map(
                                        read_labeled_unlabeled, num_parallel_calls=AUTO),
                                num_parallel_calls=tf.data.AUTOTUNE)
    
    if repeat and not full_model:
        ds = ds.repeat()
    
    if shuffle and not full_model: 
        ds = ds.shuffle(1024*8)
        opt = tf.data.Options()
        opt.experimental_deterministic = False
        ds = ds.with_options(opt)
    
    ds = ds.map(lambda image, mask_or_image_name: (images.prepare_image(image,
                                                                        dim=dim,
                                                                        seg_datasets=True),
                                                   mask_or_image_name),
                num_parallel_calls=AUTO)
    # for image dataset
    if labeled: 
        ds = ds.map(lambda image, mask: (image, images.prepare_image(mask, dim=dim,
                                                                     seg_datasets=True,
                                                                     mask=True,)),
                                                                     #mask_channels=mask_channels)),
                    num_parallel_calls=AUTO)
        
        if augment: # I am not using Test Time Augmentation, this is for Train only.
            ds = ds.flat_map(lambda image, mask: images.seg_augmenter(image, mask))
            ds = ds.shuffle(1024*8)
            
    ds = ds.batch(batch_size * config.REPLICAS, drop_remainder=True)
    ds = ds.prefetch(AUTO)
    return ds