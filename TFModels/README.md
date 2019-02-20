# Tensorflow Models

Trained tensorflow models live here. They worked hard today. Let them rest a little.

### How to import models into Tensorboard:
From your version of tensorflow, verify that the file import_pb_to_tensorboard.py lives in (PATH/TO/tensorflow/python/tools).

From there, import the model with:

```
$ python import_pb_to_tensorboard.py --model_dir /PATH/TO/<model>.pb --log_dir /tmp/tensorflow_logdir
```

Once the file is imported, view the model with ```tensorboard --logdir=/tmp/tensorflow_logdir``` to start TensorBoard.

View the model in browser with http://localhost:6006 