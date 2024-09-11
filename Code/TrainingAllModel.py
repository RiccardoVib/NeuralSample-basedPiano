import os
import tensorflow as tf
from NFLossFunctions import STFT_loss, NMSELoss
from DatasetsClass import DataGeneratorPickles
from PianoModel import create_model
from UtilsForTrainings import plotTraining, writeResults, checkpoints, render_results, MyLRScheduler
import random
import numpy as np

def train(data_dir, **kwargs):
    learning_rate = kwargs.get('learning_rate', 3e-4)
    model_save_dir = kwargs.get('model_save_dir', '../../../TrainedModels')
    save_folder = kwargs.get('save_folder', 'ED_Testing')
    batch_size = kwargs.get('batch_size', None)
    inference = kwargs.get('inference', False)
    filename = kwargs.get('filename', '')
    model_type = kwargs.get('model_type', '')
    epochs = kwargs.get('epochs', 1)

    num_steps = 1
    # tf.keras.backend.set_floatx('float64')
    #####seed
    np.random.seed(422)
    tf.random.set_seed(422)
    random.seed(422)
    fs = 48000

    gpu = tf.config.experimental.list_physical_devices('GPU')
    if len(gpu) != 0:
        tf.config.experimental.set_memory_growth(gpu[0], True)
    # tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=18000)])
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    train_gen = DataGeneratorPickles(filename, data_dir, set='train', steps=num_steps, model=None, batch_size=batch_size)

    training_steps = train_gen.lim*9


    opt = tf.keras.optimizers.Adam(learning_rate=MyLRScheduler(learning_rate, training_steps))

    model = create_model(1, num_steps, 16, b_size=batch_size, model_type=model_type)


    model.compile(loss='mse', optimizer=opt)

    print('learning_rate:', learning_rate)
    print('\n')
    print('training_steps:', training_steps)
    print('\n')

    train_gen = DataGeneratorPickles(filename, data_dir, set='train', steps=num_steps, model=model,
                                     batch_size=batch_size)
    test_gen = DataGeneratorPickles(filename, data_dir, set='val', steps=num_steps, model=model,
                                    batch_size=batch_size)

    callbacks = []
    ckpt_callback, ckpt_callback_latest, ckpt_dir, ckpt_dir_latest = checkpoints(model_save_dir, save_folder, '')
    callbacks += [ckpt_callback, ckpt_callback_latest]
    if not inference:
        last = tf.train.latest_checkpoint(ckpt_dir_latest)
        if last is not None:
            print("Restored weights from {}".format(ckpt_dir_latest))
            model.load_weights(last).expect_partial()
            #start_epoch = int(latest.split('-')[-1].split('.')[0])
            #print('Starting from epoch: ', start_epoch + 1)
        else:
            print("Initializing random weights.")

        loss_training = np.empty(epochs)
        loss_val = np.empty(epochs)
        best_loss = 1e9
        count = 0
        for i in range(epochs):
            print('epoch: ', i)
            model.reset_states()
            model.layers[2].reset()

            results = model.fit(train_gen,
                                shuffle=False,
                                validation_data=test_gen,
                                epochs=1,
                                verbose=0,
                                callbacks=callbacks)
            print(model.optimizer.learning_rate)
            loss_training[i] = (results.history['loss'])[-1]
            loss_val[i] = (results.history['val_loss'])[-1]
            if results.history['val_loss'][-1] < best_loss:
                best_loss = results.history['val_loss'][-1]
                count = 0
            else:
                count = count + 1
                if count == 20:
                    break

        writeResults(results, batch_size, learning_rate, model_save_dir, save_folder, 1)
        plotTraining(loss_training[:i], loss_val[:i], model_save_dir, save_folder, str(epochs))

    best = tf.train.latest_checkpoint(ckpt_dir)
    if best is not None:
        print("Restored weights from {}".format(ckpt_dir))
        model.load_weights(best).expect_partial()

    model.reset_states()
    model.layers[2].reset()
    test_loss = model.evaluate(test_gen,
                               verbose=0,
                               return_dict=True)
    results = {'test_loss': test_loss}
    model.reset_states()
    model.layers[2].reset()
    pred = model.predict(test_gen, verbose=0)
    render_results(pred, test_gen.x, test_gen.y, model_save_dir, save_folder, fs)

    with open(os.path.normpath('/'.join([model_save_dir, save_folder, 'results.txt'])), 'w') as f:
        for key, value in results.items():
            print('\n', key, '  : ', value, file=f)

    return 42
