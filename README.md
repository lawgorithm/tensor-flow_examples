# tensor-flow_examples

* Basic tensorflow
  * Tensorflow_introduction.ipynb
  * Sequential API vs Functional API
    * Sequential API only allows for sequential models ("linear graphs"), whereas Functional can handle other topologies (skip connections, multiple towers, shared layers etc
    * Functional API: layers are like functions -- you pass the output of a previous layer (or the input layer) as its argument. Input size matters: Because you're explicitly connecting layers, you must ensure that the output shape of a previous layer matches the input shape expected by the next layer
    * Sequential API: first layer needs to have the input shape defined, but after that, keras infers the input shape from the output shape of the prior layer. Main focus is the "type" of layer and defining the output size (number of units).

* tf.Dataset
  * tf.data.Dataset.from_tensor_slices()
    * Tensorflow_introduction.ipynb
  * tf.data.Dataset.zip()
    * Tensorflow_introduction.ipynb
  * Looking at training examples from the dataset
    * for _ in train_dataset.take(1):
      * Transfer_learning_with_MobileNet_v1.ipynb
    * next(iter(train_dataset))
      * Transfer_learning_with_MobileNet_v1.ipynb
* CNN
  * Sequential API models with tensorflow.keras.layers pre-implemented layers
    * Convolution_model_Application.ipynb
  * Functional API models
    * Residual_Networks_2022_09_17_15_50_38.ipynb
      * Functional API allows for skip connections in very deep CNNs
  * Defining datasets
    * tf.keras.preprocessing.image_dataset_from_directory()
      * Transfer_learning_with_MobileNet_v1.ipynb
  * Data augmentation
    * Transfer_learning_with_MobileNet_v1.ipynb
  * Transfer learning
    * tf.keras.applications.MobileNetV2()
      * Transfer_learning_with_MobileNet_v1.ipynb
    * Freezing layers/models: model.trainable = False and/or layer.trainable = False
      * Transfer_learning_with_MobileNet_v1.ipynb
* customizing model.compile()
  * Learning rate: optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate)
    * Transfer_learning_with_MobileNet_v1.ipynb
* customizing model.fit()
  * Learning rate: optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate)
  * Specify a validation set: validation_data=validation_dataset
  * Specify the number of epochs: epochs=K
  * Save history: history = model.fit(...)


