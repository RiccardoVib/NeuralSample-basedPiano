# Neural Sampled-based Piano Synthesis

This code repository is for the article _Neural Sampled-based Piano Synthesis_, Proceedings of the International Conference on Digital Audio Effects (DAFx25).

This repository contains all the necessary utilities to use our architectures. Find the code located inside the "./Code" folder, and the weights of pre-trained models inside the "./Weights" folder

Visit our [companion page with audio examples](https://riccardovib.github.io/NeuralSample-basedPiano_pages/)

### Folder Structure

```
./
├── Code
└── Weights
    ├── T1
    │   ├── LSTM
    │   └── S6
    └── T2
        ├── LSTM
        └── S6
```

### Contents

1. [Datasets](#datasets)
2. [How to Train and Run Inference](#how-to-train-and-run-inference)
3. [VST Download](#vst-download)

<br/>

# Datasets

Datasets are available [here](https://www.kaggle.com/datasets/riccardosimionato/pianorecordingssinglenotes/versions/2)

Our architectures were evaluated on two type of piano: Upright and Grand piano. 


# How To Train and Run Inference 

This code relies on TensorFlow.
First, install Python dependencies:
```
cd ./code
pip install -r requirements.txt
```

To train models, use the starter.py script.
Ensure you have loaded the dataset into the chosen datasets folder

Available options: 
* --model_save_dir - Folder directory in which to store the trained models [str] (default ="./models")
* --data_dir - Folder directory in which the datasets are stored [str] (default="./datasets")
* --datasets - The names of the datasets to use. [ [str] ] (default=[" "] )
* --epochs - Number of training epochs. [int] (default=60)
* --model_type - The name of the model to train ('S6', 'LSTM') [str] (default=" ")
* --cond_dim - The dimension of the conditioning vector [int] (default=1)
* --batch_size - The size of each batch [int] (default=512)
* --mini_batch_size - Number of samples to process each iteration [int] (default=512)
* --units = The hidden layer size (number of units) of the network. [ [int] ] (default=64)
* --learning_rate - the initial learning rate [float] (default=3e-4)
* --only_inference - When True, skips training and runs only inference on the pre-model. When False, runs training and inference on the trained model. [bool] (default=False)
 

Example training case: 
```
cd ./code/

python starter.py --datasets DatasetSingleNoteFilter_ --model S6 --epochs 500
```

To only run inference on an existing pre-trained model, use the "only_inference". In this case, ensure you have the existing model and dataset (to use for inference) both in their respective directories with corresponding names.

Example inference case:
```
cd ./code/

python starter.py --datasets DatasetSingleNoteFilter_ --model S6 --only_inference True
```

# VST Download

Coming soon...


# Bibtex

If you use the code included in this repository or any part of it, please acknowledge 
its authors by adding a reference to these publications:

```

@inproceedings{simionato_neuralpcm_2025,
	author = {Simionato Riccardo and Fasciani Stefano},
	title = {Neural Sampled-based Piano Synthesis},
	booktitle = {Proceedings of the International Conference on Digital Audio Effects (DAFx25)},
	year = {2025},
    address = {Ancona, Italy},
}
```
