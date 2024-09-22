from TrainingAllModel import train

"""
main script

"""

DATA_DIR = '../../Files/All/'  #### Dataset folder
MODEL_SAVE_DIR = '../../TrainedModels'  #### Models folder
INFERENCE = False ### if no training needed
STEPS = 1 ### number of timesteps per iteration
LR = 3e-4 ### initial learning rate
batch_size = 512

keys = ['A0', 'B1', 'C2', 'D3', 'E4', 'F5', 'G6', 'A#7']

models = ['LSTM', 'S6']

for key in keys:
    for model in models:
        filename = 'DatasetSingleNoteFilter_' + key

        MODEL_NAME = filename + '_' + model + ''  #### Model name

        train(data_dir=DATA_DIR,
              filename=filename,
              save_folder=MODEL_NAME,
              model_save_dir=MODEL_SAVE_DIR,
              learning_rate=LR,
              epochs=1000,
              model_type=model,
              batch_size=batch_size,
              num_steps=STEPS,
              inference=INFERENCE)
