# Project template

### Modules

* `modules.py`: Contains the pytorch modules used to build the model to be trained.
* `mnist_dataset.py` A pytorch dataset to handle MNIST images and load them directly on GPU or CPU.
* `data_ingredient.py`: Sacred ingredient which sets up data loaders.
* `model_ingredient.py`: Sacred ingredient which instantiates the model.

### Training

The `train.py` script is a `sacred` experiment which:

  * Loads the model and data ingredients
  * Creates the optimizer and trains the model

For default configuration, run as:
`python train.py -F runs -C sys`

* The `-F runs` option creates a `FileStorageObserver` which saves experiment data under the `runs` directory.
* The `-C sys` option is just a detail necessary so that the saved stdout is tidy.

To add config updates, use the standard sacred `with` keyword. For example, to run for 200 epochs, and with a batch size of 128:

`python train.py -F runs -C sys with epochs=200 dataset.batch_size=128`

To save experiment data to a MongoDB database as well as local file storage, setup an authentication spec as a json file (say `auth.json`) and add the `-M auth.json` option.

`python train.py -F runs -C sys -M auth.json`

The authentication spec should have the following structure:

```
{
  "client_kwargs":{
    "username":"<username>",
    "password":"<password>",
    "host":"<ip address of host>",
    "port":<port at which mongodb is exposed (usually 27017 by default)>
  },
  "db_name":"sacred"
}
```

# Analyzing the trained model

The analysis is done by loading saved sacred data and model checkpoints into a jupyter notebook `analysis.ipynb` using the helper package `incense`.
