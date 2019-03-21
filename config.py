import os
import logging


class Config(object):
    IMAGE_SIZE = 200

    # Should change trials to 10
    TRIALS = 20
    BATCH_SIZE = 16

    EPOCHS = 500
    PATIENCE = 100
    SAMPLES_VALIDATION = 300
    VALIDATION_SPLIT = 0.2
    TEST_SPLIT = 0.1

    DEVELOPMENT = True
    DEBUG = True
    PRINT_SQL = False
    SECRET = "example secret key"
    LOG_LEVEL = logging.DEBUG

    RAW_NRRD_ROOT = "/data/Dropbox/Dropbox/renal tumor/renal tumor/DICOM/"
    RAW_FEATURES = [
        "/data/Dropbox/Dropbox/Updated-CSV/Kideny__Malignant__clinical.csv",
        "/data/Dropbox/Dropbox/Updated-CSV/Kideny__Malignant__institutions.csv",
        "/data/Dropbox/Dropbox/Updated-CSV/Kideny__Malignant__outcomes.csv",
        ]

    DATA = "/data/intrepidlemon/renal"
    PREPROCESSED_DIR = os.path.join(DATA, "preprocessed")
    TRAIN_DIR = os.path.join(DATA, "train")
    TEST_DIR = os.path.join(DATA, "test")
    VALIDATION_DIR = os.path.join(DATA, "validation")

    FEATURES_DIR = "features"
    NRRD_FEATURES = os.path.join(FEATURES_DIR, "nrrd-features.pkl")
    FEATURES = os.path.join(FEATURES_DIR, "training-features.pkl")
    PREPROCESS = os.path.join(FEATURES_DIR, "preprocess.pkl")

    INPUT_FORM = "all"

    OUTPUT = "output"
    DB_URL = "sqlite:///{}/results.db".format(OUTPUT)
    MODEL_DIR = os.path.join(OUTPUT, "models")
    STDOUT_DIR = os.path.join(OUTPUT, "stdout")
    STDERR_DIR = os.path.join(OUTPUT, "stderr")
    DATASET_RECORDS = os.path.join(OUTPUT, "datasets")

    MAIN_TEST_HOLDOUT = 0.2
    NUMBER_OF_FOLDS = 4
    SPLIT_TRAINING_INTO_VALIDATION = 0.1


config = Config()
