# tensor-flow_examples

* __Basic tensorflow__
  * Tensorflow_introduction.ipynb
  * Sequential API vs Functional API
    * Sequential API only allows for sequential models ("linear graphs"), whereas Functional can handle other topologies (skip connections, multiple towers, shared layers etc
    * Functional API: layers are like functions -- you pass the output of a previous layer (or the input layer) as its argument. Input size matters: Because you're explicitly connecting layers, you must ensure that the output shape of a previous layer matches the input shape expected by the next layer
    * Sequential API: first layer needs to have the input shape defined, but after that, keras infers the input shape from the output shape of the prior layer. Main focus is the "type" of layer and defining the output size (number of units).

* __tf.Dataset__
  * tf.data.Dataset.from_tensor_slices()
    * Tensorflow_introduction.ipynb
    * Image_segmentation_Unet_v2.ipynb
  * tf.data.Dataset.zip()
    * Tensorflow_introduction.ipynb
  * dataset.map()
    * Image_segmentation_Unet_v2.ipynb
    * C4_W2_Lab_1_features_and_labels.ipynb
    * C2_W4_Lab_2_Signals.ipynb
  * Looking at training examples from the dataset
    * for _ in train_dataset.take(1):
      * Transfer_learning_with_MobileNet_v1.ipynb
      * C2_W4_Lab_2_Signals.ipynb
    * next(iter(train_dataset))
      * Transfer_learning_with_MobileNet_v1.ipynb
  * tf.data.Dataset.range()
    * A range of values.
    * C4_W2_Lab_1_features_and_labels.ipynb
  * dataset.window(...)
    * Windowing a dataset
    * C4_W2_Lab_1_features_and_labels.ipynb
    * C4W4_Assignment.ipynb
    * C2_W4_Lab_2_Signals.ipynb
  * dataset.shuffle()
    * Shuffling a dataset
    * C4_W2_Lab_1_features_and_labels.ipynb
  * dataset.batch().prefetch()
    * Batching and prefetching data
    * C4_W2_Lab_1_features_and_labels.ipynb
    * C2_W4_Lab_2_Signals.ipynb
  * dataset.repeat()
    * C2_W4_Lab_2_Signals.ipynb

* __Other tensorflow data types__
  * tf.data.TFRecordDataset()
    * C2W2_Assignment.ipynb
    * C2_W4_Lab_2_Signals.ipynb
    * C2_W4_Lab_3_Images.ipynb
  * tf.train.Example()
    * C2W2_Assignment.ipynb
    * C2_W4_Lab_3_Images.ipynb

* __MLP__
  * Sequential API
    * C1_W2_Lab_1_beyond_hello_world.ipynb
    * C3_W1_Lab_1_Keras_Tuner.ipynb
    * With embeddings layer:
      * C3_W2_Lab_1_imdb.ipynb
      * C3W2_Assignment.ipynb
  * Functional API
    * C3_W2_Lab_1_Manual_Dimensionality.ipynb
     
* __Image processing__
  * Add bounding boxes to common objects with detect_common_objects() in cvlib. Includes a confidence level.
    * client.ipynb
  * Read image as numpy array with cv2.imread(img_filepath)
    * client.ipynb
  * Display an image in a notebook with display(Image(filename)) from IPython.display
    * C1W2_Ungraded_Lab_Birds_Cats_Dogs.ipynb
  * img_to_array, array_to_img, load_img in tensorflow.keras.preprocessing.image
    * For loading images in a particular size and converting back and forth between the image and numpy arrays.
    * C1W2_Ungraded_Lab_Birds_Cats_Dogs.ipynb
  * display.display(display.Image(data=image_raw))
    * C2_W4_Lab_3_Images.ipynb

* __CNN__
  * Sequential API models with tensorflow.keras.layers pre-implemented layers
    * Convolution_model_Application.ipynb
    * C1_W3_Lab_1_improving_accuracy_using_convolutions.ipynb
    * C1W3_Assignment.ipynb
    * C1W4_Assignment.ipynb
    * C2_W1_Lab_1_cats_vs_dogs.ipynb
    * C2W1_Assignment.ipynb
    * C2_W4_Lab_1_multi_class_classifier.ipynb
      * Multiclass classification
    * C2W4_Assignment.ipynb
    * C1W2_Ungraded_Lab_Birds_Cats_Dogs.ipynb
  * Functional API models
    * Residual_Networks_2022_09_17_15_50_38.ipynb
      * Functional API allows for skip connections in very deep CNNs
    * Image_segmentation_Unet_v2.ipynb
      * More skip connections in a "U-shaped" pattern. U-Net is a very popular choice for  semantic segmentation tasks. U-Net has skip connections to retain information that gets lost during encoding. Skip connections send information to every upsampling layer in the decoder from the corresponding downsampling layer in the encoder.
  * Defining datasets
    * tf.keras.preprocessing.image_dataset_from_directory()
      * Generates a tf.data.Dataset from image files organized in a directory structure
      * Transfer_learning_with_MobileNet_v1.ipynb
    * tf.keras.preprocessing.image.load_img()
      * Loads a single image from a specified file path, returning a PIL object. Typically used as building block in a custom preprocessing pipeline or when you need to work with images one at a time.
      * Face_Recognition.ipynb
  * Data augmentation
    * Transfer_learning_with_MobileNet_v1.ipynb
  * Visualizing convolutions
    * C1_W3_Lab_1_improving_accuracy_using_convolutions.ipynb
  * ImageDataGenerator()
    * Flow images from a directory in batches using a ImageDataGenerator(). Can be used to rescale, compress, do data augmentation, etc on the fly as they flow during training.
    * C1_W4_Lab_1_image_generator_with_validation.ipynb
    * C1_W4_Lab_3_compacted_images.ipynb
    * C1W4_Assignment.ipynb
    * C2_W1_Lab_1_cats_vs_dogs.ipynb
    * C2W1_Assignment.ipynb
    * For data augmentation:
      * C2_W2_Lab_1_cats_v_dogs_augmentation.ipynb
      * C2_W2_Lab_2_horses_v_humans_augmentation.ipynb
      * C2W2_Assignment.ipynb
    * C2_W4_Lab_1_multi_class_classifier.ipynb
    * C2W4_Assignment.ipynb
    * C1W2_Ungraded_Lab_Birds_Cats_Dogs.ipynb

* __Transfer learning__
  * tf.keras.applications.MobileNetV2()
    * Transfer_learning_with_MobileNet_v1.ipynb
  * Freezing layers/models: model.trainable = False and/or layer.trainable = False
    * Transfer_learning_with_MobileNet_v1.ipynb
  * Loading face verification model using model_from_json() with weights in .h5 file
    * Face_Recognition.ipynb
  * Fine-tuning a pre-trained tensorflow.keras.applications.inception_v3 model for CNN. Also, freeze some of the layers.
    * C2_W3_Lab_1_transfer_learning.ipynb
    * C2W3_Assignment.ipynb
  * tf.keras.applications.VGG19()
    * Image classification, feature extraction for computer vision tasks, etc
    * Art_Generation_with_Neural_Style_Transfer.ipynb
  * Working with pre-trained word embeddings
    * Operations_on_word_vectors_v2a.ipynb
    * Emoji_v3a.ipynb
    * Embedding_plus_Positional_encoding.ipynb
    * C3W3_Assignment.ipynb
      * GloVe word embeddings
  * Defining an Embedding() layer with pre-trained embeddings
    * Emoji_v3a.ipynb
    * C3W3_Assignment.ipynb
  * Pretrained transformer model and transformer tokenizer, TFDistilBertForTokenClassification.from_pretrained()
    * Transformer_application_Named_Entity_Recognition.ipynb
  * TFDistilBertForQuestionAnswering.from_pretrained()
    * QA_dataset.ipynb

* __NLP Preprocessing and other tools__
  * Tokenizing with tensorflow.keras.preprocessing.text.Tokenizer()
    * C3_W1_Lab_1_tokenize_basic.ipynb
    * C3_W1_Lab_3_sarcasm.ipynb
    * C3W1_Assignment.ipynb
    * C3W2_Assignment.ipynb
  * pad_sequences()
    * C3_W1_Lab_2_sequences_basic.ipynb
    * C3_W1_Lab_3_sarcasm.ipynb
    * C3W1_Assignment.ipynb
    * C3W2_Assignment.ipynb
  * tf.keras.layers.Embedding()
    * C3_W2_Lab_1_imdb.ipynb
    * C3_W2_Lab_2_sarcasm_classifier.ipynb
    * C3_W2_Lab_3_imdb_subwords.ipynb
    * C3W2_Assignment.ipynb
  * n-grams
    * C3W4_Assignment.ipynb
    * C3_W4_Lab_2_irish_lyrics.ipynb

* __LSTM__
  * Sequential API models with tensorflow.keras.layers pre-implemented layers
    * tf.keras.layers.Bidirectional(tf.keras.layers.LSTM())
    * C3_W3_Lab_1_single_layer_LSTM.ipynb
    * C3_W3_Lab_2_multiple_layer_LSTM.ipynb
    * C3_W3_Lab_3_Conv1D.ipynb
    * C3_W3_Lab_4_imdb_reviews_with_GRU_LSTM_Conv1D.ipynb
    * C3_W3_Lab_5_sarcasm_with_bi_LSTM.ipynb
    * C3W3_Assignment.ipynb
    * C3W4_Assignment.ipynb
    * C3_W4_Lab_1.ipynb
    * C3_W4_Lab_2_irish_lyrics.ipynb
  * Functional API models
    * Improvise_a_Jazz_Solo_with_an_LSTM_Network_v4.ipynb
    * Emoji_v3a.ipynb
    * Neural_machine_translation_with_attention_v4a.ipynb

* __Other sequence models__
  * GRU models (Sequential API)
    * C3_W3_Lab_4_imdb_reviews_with_GRU_LSTM_Conv1D.ipynb
    * C3W3_Assignment.ipynb
  * RNN models (Sequential API)
    * tf.keras.layers.SimpleRNN()
    * C4_W3_Lab_1_RNN.ipynb
  * Conv1D()
    * Can be used for sequential data, where the convolutions happen over a single dimension (as oppposed to image data where you use 2D convolutions). Often combined with GlobalMaxPooling1D() layers to reduce dimensionality before passing to a Dense layer. Can also be used in conjunction with, for example, LSTM layers
    * Trigger_word_detection_v2a.ipynb
    * C3_W3_Lab_4_imdb_reviews_with_GRU_LSTM_Conv1D.ipynb
    * C4_W4_Lab_1_LSTM.ipynb
    * C4_W4_Lab_3_Sunspots_CNN_RNN_DNN.ipynb
    * C4W4_Assignment.ipynb

* __Attention mechanisms__
  * Neural_machine_translation_with_attention_v4a.ipynb

* __Transformers__
  * C5_W4_A1_Transformer_Subclass_v1.ipynb
  * Transformer_application_Named_Entity_Recognition.ipynb

* __customizing model.compile()__
  * Learning rate: optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate)
    * Transfer_learning_with_MobileNet_v1.ipynb
  * Dynamically decay learning rate as epochs increase
    * tf.keras.optimizers.schedules.ExponentialDecay()
    * C4_W4_Lab_3_Sunspots_CNN_RNN_DNN.ipynb

* __customizing model.fit()__
  * Learning rate: optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate)
  * Specify a validation set: validation_data=validation_dataset
  * Specify the number of epochs: epochs=K
  * Save history: history = model.fit(...)
 
* __Test set evaluation__
  * model.evaluate(..., return_dict=True)
    * C3_W1_Lab_1_Keras_Tuner.ipynb

* __masks__
  * tf.boolean_mask()
    * Autonomous_driving_application_Car_detection.ipynb

* __Loss functions__
  * Ordinary loss functions
    * categorical_crossentropy
      * Emoji_v3a.ipynb
    * binary_crossentropy
      * Trigger_word_detection_v2a.ipynb
    * sparse_categorical_cross_entropy
      * C1_W2_Lab_1_beyond_hello_world.ipynb
    * tf.keras.losses.Huber()
      * C4_W3_Lab_2_LSTM.ipynb
    * tf.keras.losses.MeanAbsoluteError()
      * C4W4_Assignment.ipynb
  * __Custom loss functions__
    * triplet_loss(y_true, y_pred, alpha = 0.2)
      * Face_Recognition.ipynb

* __with tf.GradientTape() as tape:__
  * Art_Generation_with_Neural_Style_Transfer.ipynb
  * QA_dataset.ipynb

* __Callbacks__
  * Stop training when loss reaches a target
    * tf.keras.callbacks.Callback
    * C1_W2_Lab_1_beyond_hello_world.ipynb
    * C1_W2_Lab_2_callbacks.ipynb
    * C1W2_Assignment.ipynb
    * C1W3_Assignment.ipynb
    * C4_W4_Lab_1_LSTM.ipynb
  * Automatically tune the learning rate hyperparameter
    * tf.keras.callbacks.LearningRateScheduler()
    * C4_W2_Lab_3_deep_NN.ipynb
    * C4_W3_Lab_1_RNN.ipynb
    * C4_W3_Lab_2_LSTM.ipynb
    * C4W3_Assignment.ipynb
    * C4_W4_Lab_1_LSTM.ipynb
    * C4_W4_Lab_3_Sunspots_CNN_RNN_DNN.ipynb
    * C4W4_Assignment.ipynb
  * Stop early if the loss is not improving over k epochs
    * tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=k)
    * C3_W1_Lab_1_Keras_Tuner.ipynb
  * tfmot.sparsity.keras.UpdatePruningStep()
    * Pruning is a method for reducing model size by taking parameters that are close to 0 and forcing them to be 0, in order to better enable compressability.
    * C3_W2_Lab_3_Quantization_and_Pruning.ipynb

* __Lambda layers__
  * tf.keras.layers.Lambda()
  * Reshape input data from 2D to 3D tensor
    * C4_W3_Lab_1_RNN.ipynb
  * Scale tensor entries by a scaling factor (e.g. scale by 100)
    * C4_W3_Lab_1_RNN.ipynb
    * C4_W4_Lab_1_LSTM.ipynb
  * Apply arbitrary functions to features for feature engineering
    * C3_W2_Lab_1_Manual_Dimensionality.ipynb
   
* __Data manipulation__
  * One-hot encoding with to_categorical
    * C3W4_Assignment.ipynb
    * C2_W4_Lab_3_Images.ipynb
      * tf.one_hot()
  * Reshaping tensors
    * TODO: search for np.new_axis or something like this and probably the word reshape
    * forecast.squeeze()
      * Drop unnecessary axis
    * tf.reshape(image, (32, 32, 3))
      * C2_W4_Lab_3_Images.ipynb
  * tensorflow_transform
    * Library used for feature transformations like scale_to_0_1()
    * C2_W2_Lab_1_Simple_Feature_Engineering.ipynb
  * dataset_metadata.DatasetMetadata(schema_utils.schema_from_feature_spec(...))
    * Define a schema with a schema protobuf
    * C2_W2_Lab_1_Simple_Feature_Engineering.ipynb
  * MessageToDict()
    * Convert protobuff to python dictionary
    * C2_W2_Assignment.ipynb
   
* __Generative tasks__
  * Next-word prediction with model.predict() on an LSTM model
    * Seems pretty similar to forecasting the next observation in a time-series
    * C3W4_Assignment.ipynb
    * C3_W4_Lab_1.ipynb
    * C3_W4_Lab_2_irish_lyrics.ipynb

* __Time series__
  * Generating and plotting synthetic time series data
    * C4_W1_Lab_1_time_series.ipynb
  * Naive and simple forecasting models (not really using tensorflow)
    * C4W1_Assignment.ipynb
  * Shallow regression model using previous k observations as features (Sequential API)
    * C4_W2_Lab_2_single_layer_NN.ipynb
  * Deep MLP model using previous k observations as features (Sequential API)
    * C4_W2_Lab_3_deep_NN.ipynb
    * C4W2_Assignment.ipynb
    * C4_W4_Lab_2_Sunspots_DNN.ipynb
  * Deep RNN using previous k observations as features (Sequential API)
    * C4_W3_Lab_1_RNN.ipynb
  * Deep LSTM using previous k observations as features (Sequential API)
    * C4_W3_Lab_2_LSTM.ipynb
    * C4W3_Assignment.ipynb
    * C4_W4_Lab_1_LSTM.ipynb
    * C4_W4_Lab_3_Sunspots_CNN_RNN_DNN.ipynb
    * C4W4_Assignment.ipynb
  * Faster forecasting inference using tf.data.Dataset
    * C4W3_Assignment.ipynb
    * C4W4_Assignment.ipynb

* __Analysis__
  * Confusion matrices: from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    * C1W2_Ungraded_Lab_Birds_Cats_Dogs.ipynb
  * tfdv.generate_statistics_from_dataframe() and tfdv.visualize_statistics()
    * Plot some automated analysis of your data columns for a pandas df
    * C2_W1_Lab_1_TFDV_Exercise.ipynb
    * C2W1_Assignment.ipynb
  * tfdv.infer_schema() and tfdv.display_schema()
    * Tool for automatically inferring the schema of your data
    * C2_W1_Lab_1_TFDV_Exercise.ipynb
    * C2W1_Assignment.ipynb
  * tfdv.display_anomalies()
    * e.g. look for anomalies in the eval set (example: unexpected categories that don't appear in the training set). Basically, trying to detect a skew in the validation set relative to the training set.
    * C2_W1_Lab_1_TFDV_Exercise.ipynb
    * C2W1_Assignment.ipynb
  * 3D Plotting
    * from mpl_toolkits.mplot3d import Axes3D
      * C3_W2_Lab_2_Algorithmic_Dimensionality.ipynb

* __Hyperparameter Selection__
  * keras_tuner library
    * Several choices of algorithms for exploring a defined space of hyperparameters (BayesianOptimization, Hyperband, etc)
    * C3_W1_Lab_1_Keras_Tuner.ipynb
   
* __Distributed Training__
  * C3W3_Colab_Lab1_Distributed_Training.ipynb
 
* __Distillation with Teacher and Student Models__
  * C3_W3_Lab_2_Knowledge_Distillation.ipynb


