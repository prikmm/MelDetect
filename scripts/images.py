import tensorflow as tf
import math
import tensorflow.keras.backend as K
import numpy as np
from functools import partial

AUTO = tf.data.experimental.AUTOTUNE

def prepare_image(img, dim=256):  
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.reshape(img, [dim, dim, 3])
    return img


@tf.function
def transform_grid_mark(image, inv_mat, image_shape):
    h, w, c = image_shape
    
    cx, cy = w//2, h//2

    new_xs = tf.repeat( tf.range(-cx, cx, 1), h)
    new_ys = tf.tile( tf.range(-cy, cy, 1), [w])
    new_zs = tf.ones([h*w], dtype=tf.int32)

    old_coords = tf.matmul(inv_mat, tf.cast(tf.stack([new_xs, new_ys, new_zs]), tf.float32))
    old_coords_x, old_coords_y = tf.round(old_coords[0, :] + tf.cast(w, tf.float32)//2.), tf.round(old_coords[1, :] + tf.cast(h, tf.float32)//2.)
    old_coords_x = tf.cast(old_coords_x, tf.int32)
    old_coords_y = tf.cast(old_coords_y, tf.int32)    

    clip_mask_x = tf.logical_or(old_coords_x<0, old_coords_x>w-1)
    clip_mask_y = tf.logical_or(old_coords_y<0, old_coords_y>h-1)
    clip_mask = tf.logical_or(clip_mask_x, clip_mask_y)

    old_coords_x = tf.boolean_mask(old_coords_x, tf.logical_not(clip_mask))
    old_coords_y = tf.boolean_mask(old_coords_y, tf.logical_not(clip_mask))
    new_coords_x = tf.boolean_mask(new_xs+cx, tf.logical_not(clip_mask))
    new_coords_y = tf.boolean_mask(new_ys+cy, tf.logical_not(clip_mask))

    old_coords = tf.cast(tf.stack([old_coords_y, old_coords_x]), tf.int32)
    new_coords = tf.cast(tf.stack([new_coords_y, new_coords_x]), tf.int64)
    rotated_image_values = tf.gather_nd(image, tf.transpose(old_coords))
    rotated_image_channel = list()
    for i in range(c):
        vals = rotated_image_values[:,i]
        sparse_channel = tf.SparseTensor(tf.transpose(new_coords), vals, [h, w])
        rotated_image_channel.append(tf.sparse.to_dense(sparse_channel, default_value=0, validate_indices=False))

    return tf.transpose(tf.stack(rotated_image_channel), [1,2,0])


@tf.function
def random_rotate(image, angle, image_shape):
    def get_rotation_mat_inv(angle):
          #transform to radian
        angle = math.pi * angle / 180

        cos_val = tf.math.cos(angle)
        sin_val = tf.math.sin(angle)
        one = tf.constant([1], tf.float32)
        zero = tf.constant([0], tf.float32)

        rot_mat_inv = tf.concat([cos_val, sin_val, zero,
                                     -sin_val, cos_val, zero,
                                     zero, zero, one], axis=0)
        rot_mat_inv = tf.reshape(rot_mat_inv, [3,3])

        return rot_mat_inv
    angle = float(angle) * tf.random.normal([1],dtype='float32')
    rot_mat_inv = get_rotation_mat_inv(angle)
    return transform_grid_mark(image, rot_mat_inv, image_shape)


@tf.function
def grid_mask(DIM=384):
    h = tf.constant(DIM, dtype=tf.float32)
    w = tf.constant(DIM, dtype=tf.float32)
    
    image_height, image_width = (h, w)

    # CHANGE THESE PARAMETER
    d1 = int(DIM / 6)
    d2 = int(DIM / 4)
    rotate_angle = 45
    ratio = 0.4 # this is delete ratio, so keep ratio = 1 - delete

    hh = tf.math.ceil(tf.math.sqrt(h*h+w*w))
    hh = tf.cast(hh, tf.int32)
    hh = hh+1 if hh%2==1 else hh
    d = tf.random.uniform(shape=[], minval=d1, maxval=d2, dtype=tf.int32)
    l = tf.cast(tf.cast(d,tf.float32)*ratio+0.5, tf.int32)

    st_h = tf.random.uniform(shape=[], minval=0, maxval=d, dtype=tf.int32)
    st_w = tf.random.uniform(shape=[], minval=0, maxval=d, dtype=tf.int32)

    y_ranges = tf.range(-1 * d + st_h, -1 * d + st_h + l)
    x_ranges = tf.range(-1 * d + st_w, -1 * d + st_w + l)

    for i in range(0, hh//d+1):
        s1 = i * d + st_h
        s2 = i * d + st_w
        y_ranges = tf.concat([y_ranges, tf.range(s1,s1+l)], axis=0)
        x_ranges = tf.concat([x_ranges, tf.range(s2,s2+l)], axis=0)

    x_clip_mask = tf.logical_or(x_ranges <0 , x_ranges > hh-1)
    y_clip_mask = tf.logical_or(y_ranges <0 , y_ranges > hh-1)
    clip_mask = tf.logical_or(x_clip_mask, y_clip_mask)

    x_ranges = tf.boolean_mask(x_ranges, tf.logical_not(clip_mask))
    y_ranges = tf.boolean_mask(y_ranges, tf.logical_not(clip_mask))

    hh_ranges = tf.tile(tf.range(0,hh), [tf.cast(tf.reduce_sum(tf.ones_like(x_ranges)), tf.int32)])
    x_ranges = tf.repeat(x_ranges, hh)
    y_ranges = tf.repeat(y_ranges, hh)

    y_hh_indices = tf.transpose(tf.stack([y_ranges, hh_ranges]))
    x_hh_indices = tf.transpose(tf.stack([hh_ranges, x_ranges]))

    y_mask_sparse = tf.SparseTensor(tf.cast(y_hh_indices, tf.int64),  tf.zeros_like(y_ranges), [hh, hh])
    y_mask = tf.sparse.to_dense(y_mask_sparse, 1, False)

    x_mask_sparse = tf.SparseTensor(tf.cast(x_hh_indices, tf.int64), tf.zeros_like(x_ranges), [hh, hh])
    x_mask = tf.sparse.to_dense(x_mask_sparse, 1, False)

    mask = tf.expand_dims( tf.clip_by_value(x_mask + y_mask, 0, 1), axis=-1)

    mask = random_rotate(mask, rotate_angle, [hh, hh, 1])
    mask = tf.image.crop_to_bounding_box(mask, (hh-tf.cast(h, tf.int32))//2, (hh-tf.cast(w, tf.int32))//2, tf.cast(image_height, tf.int32), tf.cast(image_width, tf.int32))

    return mask


@tf.function
def apply_grid_mask(image, DIM=384):
    mask = grid_mask(DIM=DIM)
    mask = tf.concat([mask, mask, mask], axis=-1)
    return image * tf.cast(mask, 'float32')



def featvec_augmenter(img_specs=None, target=None, image_only=True, grid_mask=True, grid_mask_aug=False):
    position_change_func_list = [
        tf.image.stateless_random_flip_left_right,
        tf.image.stateless_random_flip_up_down,
    ]
    
    color_change_func_list =[
        partial(tf.image.stateless_random_brightness, max_delta=0.95),
        partial(tf.image.stateless_random_contrast, upper=0.5, lower=0.1),
        partial(tf.image.stateless_random_hue, max_delta=0.3),
        partial(tf.image.stateless_random_saturation, upper=0.6, lower=0.1), 
    ] 

    aug_img_array = tf.expand_dims(img_specs["img_input"], 0)
    

    for i, aug_func in enumerate(position_change_func_list):
        rand_seed = tf.random.uniform(shape=[2], minval=-10**5, maxval=10**5, dtype=tf.int64)
        aug_img = aug_func(img_specs["img_input"], seed=rand_seed)
        aug_img_array = tf.concat([aug_img_array, tf.expand_dims(aug_img, 0)], 0)
    
    for i, aug_func in enumerate(color_change_func_list):
        rand_seed = tf.random.uniform(shape=[2], minval=-10**5, maxval=10**5, dtype=tf.int64)
        aug_img = aug_func(img_specs["img_input"], seed=rand_seed)
        aug_img_array = tf.concat([aug_img_array, tf.expand_dims(aug_img, 0)], 0)
    
    if grid_mask:
        grid_masked_img = apply_grid_mask(img_specs["img_input"])
        aug_img_array = tf.concat([aug_img_array, tf.expand_dims(grid_masked_img, 0)], 0)
        if grid_mask_aug:
            all_aug_func_list = position_change_func_list + color_change_func_list
            for i, aug_func in enumerate(all_aug_func_list):
                rand_seed = tf.random.uniform(shape=[2], minval=-10**5, maxval=10**5, dtype=tf.int64)
                grid_mask_aug_img = aug_func(grid_masked_img, seed=rand_seed)
                aug_img_array = tf.concat([aug_img_array, tf.expand_dims(aug_img, 0)], 0)
        
    augmented_img_ds = tf.data.Dataset.from_tensor_slices(aug_img_array)
    target_ds = tf.data.Dataset.from_tensors(target).repeat(7)
    if image_only:
        final_aug_ds = tf.data.Dataset.zip((augmented_img_ds, target_ds))
    if not image_only:
        metadata_ds = tf.data.Dataset.from_tensors(img_specs["metadata_input"]).repeat(7)
        final_aug_ds = tf.data.Dataset.zip((augmented_img_ds, metadata_ds, target_ds))
    
    def img_only_input_handler(img, target):
        return {"img_input": img}, target
            
    def metadata_incl_input_handler(img, metadata, target):
        return {"img_input": img, "metadata_input": metadata}, target
    
    if image_only:
        get_input_handler = img_only_input_handler
    else:
        get_input_handler = metadata_incl_input_handler
        
    final_aug_ds = final_aug_ds.map(get_input_handler, num_parallel_calls=AUTO)
    return final_aug_ds



def seg_augmenter(image, mask=None, metadata=None, target=None):
    position_change_func_list = [
        tf.image.stateless_random_flip_left_right,
        tf.image.stateless_random_flip_up_down,
    ]
    
    color_change_func_list =[
        partial(tf.image.stateless_random_brightness, max_delta=0.95),
        partial(tf.image.stateless_random_contrast, upper=0.5, lower=0.1),
        partial(tf.image.stateless_random_hue, max_delta=0.3),
        partial(tf.image.stateless_random_saturation, upper=0.6, lower=0.1), 
    ] 

    #print(image["seg_input"].shape)
    img_cum_mask_array = tf.stack([image["seg_input"], mask], 3)
    #print(img_cum_mask_array.shape)
    #print(asdad)
    

    for i, aug_func in enumerate(position_change_func_list):
        rand_seed = tf.random.uniform(shape=[2], minval=-10**5, maxval=10**5, dtype=tf.int64)
        aug_img = aug_func(image["seg_input"], seed=rand_seed)
        aug_mask = aug_func(mask, seed=rand_seed)
        augmented_img_cum_mask = tf.stack([aug_img, aug_mask], 3)
        if i == 0:
            img_cum_mask_array = tf.stack([img_cum_mask_array,
                                            augmented_img_cum_mask], 0)
        else:
            img_cum_mask_array = tf.concat([img_cum_mask_array,
                                            tf.expand_dims(augmented_img_cum_mask, 0)], 0)
        #print(img_cum_mask_array.shape)
    
    for i, aug_func in enumerate(color_change_func_list):
        rand_seed = tf.random.uniform(shape=[2], minval=-10**5, maxval=10**5, dtype=tf.int64)
        aug_img = aug_func(image["seg_input"], seed=rand_seed)
        augmented_img_cum_mask = tf.stack([aug_img, mask], 3)
        img_cum_mask_array = tf.concat([img_cum_mask_array,
                                        tf.expand_dims(augmented_img_cum_mask, 0)], 0)
        #print(img_cum_mask_array.shape)


    augmented_ds = tf.data.Dataset.from_tensor_slices(img_cum_mask_array)
    
    def input_handler(img_cum_mask):
        return {"seg_input": img_cum_mask[:, :, :, 0]}, img_cum_mask[:, :, :, 1]
    
    augmented_ds = augmented_ds.map(lambda img_cum_mask: input_handler(img_cum_mask),
                                    num_parallel_calls=AUTO)
    return augmented_ds
