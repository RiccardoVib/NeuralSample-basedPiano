# Copyright (C) 2025 Riccardo Simionato, University of Oslo
# Inquiries: riccardo.simionato.vib@gmail.com.com
#
# This code is free software: you can redistribute it and/or modify it under the terms
# of the GNU Lesser General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
#
# This code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Less General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License along with this code.
# If not, see <http://www.gnu.org/licenses/>.
#
# If you use this code or any part of it in any program or publication, please acknowledge
# its authors by adding a reference to this publication:
#
# R. Simionato, 2025, "Neural Sampled-based Piano Synthesis" in proceedings of the 25th Digital Audio Effect Conference, Ancona, Italy.

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
