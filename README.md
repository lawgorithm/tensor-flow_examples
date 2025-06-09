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
  * Looking at training examples from the dataset
    * for _ in train_dataset.take(1):
      * Transfer_learning_with_MobileNet_v1.ipynb
    * next(iter(train_dataset))
      * Transfer_learning_with_MobileNet_v1.ipynb
     
* __MLP__
  * Sequential API
    * C1_W2_Lab_1_beyond_hello_world.ipynb

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
  * Conv1D()
    * Trigger_word_detection_v2a.ipynb
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
  * Defining an Embedding() layer with pre-trained embeddings
    * Emoji_v3a.ipynb
  * Pretrained transformer model and transformer tokenizer, TFDistilBertForTokenClassification.from_pretrained()
    * Transformer_application_Named_Entity_Recognition.ipynb
  * TFDistilBertForQuestionAnswering.from_pretrained()
    * QA_dataset.ipynb

* __NLP Preprocessing__
  * Tokenizing with tensorflow.keras.preprocessing.text.Tokenizer()
    * C3_W1_Lab_1_tokenize_basic.ipynb
    * C3_W1_Lab_3_sarcasm.ipynb
    * C3W1_Assignment.ipynb
  * pad_sequences()
    * C3_W1_Lab_2_sequences_basic.ipynb
    * C3_W1_Lab_3_sarcasm.ipynb
    * C3W1_Assignment.ipynb

* __LSTM__
  * Sequential API models with tensorflow.keras.layers pre-implemented layers
    * TODO
  * Functional API models
    * Improvise_a_Jazz_Solo_with_an_LSTM_Network_v4.ipynb
    * Emoji_v3a.ipynb
    * Neural_machine_translation_with_attention_v4a.ipynb

* __Attention mechanisms__
  * Neural_machine_translation_with_attention_v4a.ipynb

* __Transformers__
  * C5_W4_A1_Transformer_Subclass_v1.ipynb
  * Transformer_application_Named_Entity_Recognition.ipynb

* __customizing model.compile()__
  * Learning rate: optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate)
    * Transfer_learning_with_MobileNet_v1.ipynb

* __customizing model.fit()__
  * Learning rate: optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate)
  * Specify a validation set: validation_data=validation_dataset
  * Specify the number of epochs: epochs=K
  * Save history: history = model.fit(...)

* __Masks__
  * tf.boolean_mask()
    * Autonomous_driving_application_Car_detection.ipynb

* __Loss functions__
  * Normal loss functions
    * categorical_crossentropy
      * Emoji_v3a.ipynb
    * binary_crossentropy
      * Trigger_word_detection_v2a.ipynb
    * sparse_categorical_cross_entropy
      * C1_W2_Lab_1_beyond_hello_world.ipynb
  * __Custom loss functions__
    * triplet_loss(y_true, y_pred, alpha = 0.2)
      * Face_Recognition.ipynb

* __with tf.GradientTape() as tape:__
  * Art_Generation_with_Neural_Style_Transfer.ipynb
  * QA_dataset.ipynb

* __Callbacks__
  * Stop training when loss reaches a target
    * C1_W2_Lab_1_beyond_hello_world.ipynb
    * C1_W2_Lab_2_callbacks.ipynb
    * C1W2_Assignment.ipynb
    * C1W3_Assignment.ipynb




