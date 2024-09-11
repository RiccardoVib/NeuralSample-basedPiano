from TrainingAllModel import train

DATA_DIR = '../../Files/All/'  #### Dataset folder
MODEL_SAVE_DIR = '../../TrainedModels'  #### Models folder
INFERENCE = False
STEPS = 1
LR = 3e-4
batch_size = 512

# keys = ['C3', 'C#3', 'D3', 'D#3', 'E3', 'F3', 'F#3', 'G3', 'G#3', 'A3', 'A#3', 'B3', 'C4', 'C#4', 'D4', 'D#4', 'E4', 'F4', 'F#4', 'G4', 'G#4', 'A4', 'A#4', 'B4']

keys = ['A0', 'B1', 'C2', 'D3', 'E4', 'F5', 'G6', 'A#7']
keys = ['A0']  # 'A0',

models = ['LSTM', 'CNN', 'S6']  # 'SSM'
models = ['S6']  # 'SSM'

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
