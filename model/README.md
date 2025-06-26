# CONTENT

This folder (`model/`) contains all the files required for model instantiation, training, inference and testing.

## Model
`UNet.py` contains the model and the training functions. Important hyperparameters such as model name initial learning rate, batch size, epochs, patch shape and loss function can be modified in the first lines of this file, under the "Hyperparameters" comment. Relevant paths can also be modified under the "Paths" comment.

## Training
### Fully superivsed
The code to train the model is stored in `UNet.py`. To run training, modify hyperparameters and paths in this folder, and run the file. A progress bar will be displayed, and periodic saves of the model will be saved in the specified checkpoint folder. Logs of the training will also be saved, to allow visualization of the training process with Tensorflow. 

### Semi-supervised
To train the model in a semi-supervised manner, run analogously the code in `domain_adaptation.py` as `one_step`.

### Domain adaptation
todo: verify if this will be actually done


## Inference and testing
To run inference and verify loss, adjust the paths to the model and to the crops to be tested, and run `inference.py`.

By default, it will create a copy of the crops adding inference as `foreground` and `boundaries` datasets; it will then calculate and print intersection over union and dice loss, both per-sample and averaged.

