import tensorflow as tf

DEVICE = "TPU" #or "GPU"

# USE DIFFERENT SEED FOR DIFFERENT STRATIFIED KFOLD
SEED = 42

# NUMBER OF FOLDS. USE 3, 5, OR 15 
FOLDS = 3

# WHICH IMAGE SIZES TO LOAD EACH FOLD
# CHOOSE 128, 192, 256, 384, 512, 768 
IMG_SIZES = [384,384,384]

# INCLUDE OLD COMP DATA? YES=1 NO=0
INC2018 = [1,1,1]
INC2019 = [1,1,1]
INC2020 = [1,1,1]

# BATCH SIZE AND EPOCHS
BATCH_SIZES = [32]*FOLDS
EPOCHS = [15, 15, 15]

# WEIGHTS FOR FOLD MODELS WHEN PREDICTING TEST
WGTS = [1/FOLDS]*FOLDS

# TEST TIME AUGMENTATION STEPS
TTA = 11

# Variables that will be initialized at run time
REPLICAS = None
AUTO = None
strategy = None
DATA_PATH = None
DATA_PATH2 = None
DATA_PATH3 = None
VERBOSE = 2
DISPLAY_PLOT = True
IMG_ONLY = True
SEG_TFREC_PATHS = None
ROOT_FEATVEC_PATH = None
ROOT_ENSEMBLE_PATH = None
GRID_MASK = None
GRID_MASK_AUG = None

def tpu_gpu_initializer(DEVICE=DEVICE):
    if DEVICE == "TPU":
        print("connecting to TPU...")
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print('Running on TPU ', tpu.master())
    except ValueError:
        print("Could not connect to TPU")
        tpu = None

    if tpu:
        try:
            print("initializing  TPU ...")
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)
            strategy = tf.distribute.TPUStrategy(tpu)
            print("TPU initialized")
        except Exception:
            print("failed to initialize TPU")
    else:
        DEVICE = "GPU"

    if DEVICE != "TPU":
        print("Using default strategy for CPU and single GPU")
        strategy = tf.distribute.get_strategy()

    if DEVICE == "GPU":
        print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
        

    AUTO     = tf.data.experimental.AUTOTUNE
    REPLICAS = strategy.num_replicas_in_sync
    print(f'REPLICAS: {REPLICAS}')
    return REPLICAS, AUTO, strategy