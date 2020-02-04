# Project template

### Modules

* `modules.py`: Contains the pytorch modules used to build the model to be trained.
* `mnist_dataset.py` A pytorch dataset to handle MNIST images and load them directly on GPU or CPU.
* `training_functions.py`: Contains the `train_on_batch` and `validate` functions used in the training loop. Also has LR scheduling callbacks.
* `data_ingredient.py`: Sacred ingredient which sets up data loaders.
* `model_ingredient.py`: Sacred ingredient which instantiates the model.

### Scripts

These are sacred experiments.
* `train.py`:
  - Loads the model and data ingredients
  - Creates the optimizer and trains the model
