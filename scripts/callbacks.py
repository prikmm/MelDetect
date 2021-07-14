import tensorflow as tf
import os
import config

root_checkpointdir = os.path.join(os.curdir, "my_checkpoints")
def get_model_checkpoint_path(model_name, fold, img_size):
    file_name = model_name+"_"+"fold"+str(fold) + "_" + str(img_size)
    path = os.path.join(root_checkpointdir, model_name)
    os.makedirs(path, exist_ok=True)
    return os.path.join(path, file_name)


def get_lr_callback(batch_size=8, lr_start=0.000005, lr_min=None, lr_max=None):
    if not lr_max:
        lr_max = 0.00000125 * config.REPLICAS * batch_size
    if not lr_min:
        lr_min = 0.000001
    lr_ramp_ep = 5
    lr_sus_ep  = 0
    lr_decay   = 0.8
   
    def lrfn(epoch):
        if epoch < lr_ramp_ep:
            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
            
        elif epoch < lr_ramp_ep + lr_sus_ep:
            lr = lr_max
            
        else:
            lr = (lr_max - lr_min) * lr_decay**(epoch - lr_ramp_ep - lr_sus_ep) + lr_min
            
        return lr

    lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=False)
    return lr_callback