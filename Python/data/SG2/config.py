import torch

START_TRAIN_IMG_SIZE    = 4
DATASET                 = "/data/PROCESSED_DATASET/TESTDATASET"
ANNOTATION_PATH         = "/data/PROCESSED_DATASET/TESTDATASET/dataset.json"
DEVICE                  = "cuda" if torch.cuda.is_available() else "cpu" # run on gpu if possible
EPOCHS                  = 50
LEARNING_RATE           = 1e-3 # 0.001``
BATCH_SIZE              = 14
LOG_RESOLUTION          = 8 # 7 = 128
                            # 8 = 256
                            # 9 = 512
                            # 10 = 1024
Z_DIM                   = 512
W_DIM                   = 512
TEXT_DIM                = 512
CLIP_DIM                = 512
LAMBDA_GP               = 10 # This loss contains a parameter name λ and it's common to set λ = 10
LOAD_CHECKPOINT         = False
CHECKPOINT_PATH         = "/app/Python/data/SG2/saved_training/saved_checkpoints/TESTING!!!_2024-04-21_epoch_150.pth"
FILENAME                = "/app/Python/data/SG2/saved_training/saved_checkpoints"
MODEL_FILENAME          = "MODEL_TESTING_01"
MODEL_EXPORT_PATH       = "/app/Python/data/SG2/saved_training/saved_models"
CRITIC_ITERATIONS       = 5
PKL_PATH                = "/app/Python/data/SG2/saved_training/saved_pickles/Test_model_01_2024-04-23_epoch_150.pkl"
GENERATED_IMAGE_PATH    = "/app/Python/data/SG2/saved_training/saved_generated_images"

