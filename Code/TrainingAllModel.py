import os
import tensorflow as tf
from DatasetsClass import DataGeneratorPickles
from PianoModel import create_model
from UtilsForTrainings import plotTraining, writeResults, checkpoints, render_results, MyLRScheduler
import random
import numpy as np

def train(data_dir, **kwargs):
    """
      :param save_folder: the directory in which the models are saved [string]
      :param batch_size: the size of each batch [int]
      :param learning_rate: the initial leanring rate [float]
      :param model_save_dir: the directory in which models are stored [string]
      :param save_folder: the directory in which the model will be saved [string]
      :param inference: if True it skip the training and it compute only the inference [bool]
      :param model_type: model to use [string]
      :param dataset: name of the datset to use [string]
      :param filename: name of the dataset [string]
      :param epochs: the number of epochs [int]
    """
    
    learning_rate = kwargs.get('learning_rate', 3e-4)
    model_save_dir = kwargs.get('model_save_dir', '../../../TrainedModels')
    save_folder = kwargs.get('save_folder', 'ED_Testing')
    batch_size = kwargs.get('batch_size', None)
    inference = kwargs.get('inference', False)
    filename = kwargs.get('filename', '')
    model_type = kwargs.get('model_type', '')
    epochs = kwargs.get('epochs', 1)

    num_steps = 1
    fs = 48000

    # tf.keras.backend.set_floatx('float64')

    # set all the seed in case reproducibility is desired
    #np.random.seed(422)
    #tf.random.set_seed(422)
    #random.seed(422)

    # check if GPUs are available and set the memory growing
    gpu = tf.config.experimental.list_physical_devices('GPU')
    if len(gpu) != 0:
        tf.config.experimental.set_memory_growth(gpu[0], True)
    # tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=18000)])
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # create the DataGenerator object to retrive the data in the test set
    train_gen = DataGeneratorPickles(filename, data_dir, set='train', steps=num_steps, model=None, batch_size=batch_size)
    
    # the number of total training steps
    training_steps = train_gen.lim*9
    # define the Adam optimizer with initial learning rate, training steps
    opt = tf.keras.optimizers.Adam(learning_rate=MyLRScheduler(learning_rate, training_steps))
    
    # create the model
    model = create_model(1, num_steps, 16, b_size=batch_size, model_type=model_type)
    # compile the model
    model.compile(loss='mse', optimizer=opt)

    print('learning_rate:', learning_rate)
    print('\n')
    print('training_steps:', training_steps)
    print('\n')

    train_gen = DataGeneratorPickles(filename, data_dir, set='train', steps=num_steps, model=model,
                                     batch_size=batch_size)
    test_gen = DataGeneratorPickles(filename, data_dir, set='val', steps=num_steps, model=model,
                                    batch_size=batch_size)
    
    # define callbacks: where to store the weights
    callbacks = []
    ckpt_callback, ckpt_callback_latest, ckpt_dir, ckpt_dir_latest = checkpoints(model_save_dir, save_folder, '')
    callbacks += [ckpt_callback, ckpt_callback_latest]
    
    # if inference is True, it jump directly to the inference section without train the model
    if not inference:
        last = tf.train.latest_checkpoint(ckpt_dir_latest)
        if last is not None:
            print("Restored weights from {}".format(ckpt_dir_latest))
            # load the weights of the last epoch, if any
            model.load_weights(last).expect_partial()
      
        else:
            # if no weights are found,the weights are random generated
            print("Initializing random weights.")

        # defining the array taking the training and validation losses
        loss_training = np.empty(epochs)
        loss_val = np.empty(epochs)
        best_loss = 1e9
        # counting for early stopping
        count = 0
        
        # training loop
        for i in range(epochs):
            print('epoch: ', i)
            # reset the model's states
            model.reset_states()

            results = model.fit(train_gen,
                                shuffle=False,
                                validation_data=test_gen,
                                epochs=1,
                                verbose=0,
                                callbacks=callbacks)
            print(model.optimizer.learning_rate)
            # store the training and validation loss
            loss_training[i] = (results.history['loss'])[-1]
            loss_val[i] = (results.history['val_loss'])[-1]
            # if validation loss is smaller then the best loss, the early stopping counting is reset
            if results.history['val_loss'][-1] < best_loss:
                best_loss = results.history['val_loss'][-1]
                count = 0
            else:
                # if not count is increased by one and if equal to 20 the training is stopped
                count = count + 1
                if count == 20:
                    break
        # write and save results
        writeResults(results, batch_size, learning_rate, model_save_dir, save_folder, 1)
        # plot the training and validation loss for all the training
        plotTraining(loss_training[:i], loss_val[:i], model_save_dir, save_folder, str(epochs))

    # load the best weights of the model
    best = tf.train.latest_checkpoint(ckpt_dir)
    if best is not None:
        print("Restored weights from {}".format(ckpt_dir))
        model.load_weights(best).expect_partial()
    else:
        # if no weights are found, there is something wrong
        print("Something is wrong.")
        
    # reset the states before predicting
    model.reset_states()
    test_loss = model.evaluate(test_gen,
                               verbose=0,
                               return_dict=True)
    results = {'test_loss': test_loss}
    model.reset_states()
    # predict the test set
    pred = model.predict(test_gen, verbose=0)

    # plot and render the output audio file, together with the input and target
    render_results(pred, test_gen.x, test_gen.y, model_save_dir, save_folder, fs)
    
    # writhe and store the metrics values
    with open(os.path.normpath('/'.join([model_save_dir, save_folder, 'results.txt'])), 'w') as f:
        for key, value in results.items():
            print('\n', key, '  : ', value, file=f)

    return 42
