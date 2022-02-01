import tensorflow as tf
import tensorflow_datasets as tfds
import random
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import seaborn as sns
# !pip install adversarial-robustness-toolbox
from art.estimators.classification import TensorFlowV2Classifier, EnsembleClassifier
from art.attacks.evasion import AdversarialPatch
# !pip install tensorflow_addons
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import random




img_height = 500
img_width = 500
batch_size = 32
ds_train = tf.keras.preprocessing.image_dataset_from_directory(
    "/home/pbenga2s/RnD/Beans_OOD/Beans_OOD/train",
    labels="inferred",
    label_mode="categorical",  
    color_mode="rgb",
    batch_size=batch_size,
    image_size=(img_height, img_width), 
    shuffle=True,
    seed=123,
)
ds_test = tf.keras.preprocessing.image_dataset_from_directory(
    "/home/pbenga2s/RnD/Beans_OOD/Beans_OOD/validation",
    labels="inferred",
    label_mode="categorical",
    color_mode="rgb",
    batch_size=batch_size,
    image_size=(img_height, img_width),
    shuffle=True,
    seed=123,
)



class_names = np.array(ds_train.class_names)
print(len(class_names))
normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
ds_train = ds_train.map(lambda x, y: (normalization_layer(x), y))
AUTOTUNE = tf.data.AUTOTUNE
ds_train = ds_train.shuffle(42,reshuffle_each_iteration=False)
ds_train = ds_train.cache().prefetch(buffer_size=AUTOTUNE)
normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
ds_test = ds_test.map(lambda x, y: (normalization_layer(x), y))
AUTOTUNE = tf.data.AUTOTUNE
ds_test = ds_test.shuffle(42,reshuffle_each_iteration=False)
ds_test = ds_test.cache().prefetch(buffer_size=AUTOTUNE)


model=tf.keras.applications.MobileNetV2(
    input_shape=(500,500,3), alpha=0.35,include_top=False,
    weights='imagenet',  classes=4
)
model.trainable = True
model.summary()

image_batch, label_batch = next(iter(ds_train))
feature_batch = model(image_batch)
print(feature_batch.shape)
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)
prediction_layer = tf.keras.layers.Dense(4)
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)
preprocess_input = tf.keras.applications.mobilenet.preprocess_input
data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip("horizontal"),
        layers.experimental.preprocessing.RandomRotation(0.1),
    ]
)

inputs = tf.keras.Input(shape=(500, 500, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = model(x, training=True)
x = global_average_layer(x)
x = layers.Dense(units=1280, activation="relu")(x)
x = layers.Dropout(0.3)(x)
# x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(0.0001),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(
    ds_train,
    epochs=100,
    validation_data=ds_test,
)

model.evaluate(ds_test)

export_path = "/home/pbenga2s/RnD/Beans_OOD/saved_models/MobileNet/MobileNet_model(100)"
model.save(export_path)

export_path