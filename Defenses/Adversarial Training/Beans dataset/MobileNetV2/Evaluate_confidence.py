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
import pandas as pd





img_height = 500
img_width = 500
batch_size = 32
ds_train = tf.keras.preprocessing.image_dataset_from_directory(
    "/home/pbenga2s/RnD/Beans_advdef/MobileNetV2/Beans/train",
    labels="inferred",
    label_mode="categorical",  
    color_mode="rgb",
    batch_size=batch_size,
    image_size=(img_height, img_width), 
    shuffle=True,
    seed=123,
)
ds_test = tf.keras.preprocessing.image_dataset_from_directory(
    "/home/pbenga2s/RnD/Beans_advdef/MobileNetV2/Beans/validation",
    labels="inferred",
    label_mode="categorical",
    color_mode="rgb",
    batch_size=batch_size,
    image_size=(img_height, img_width),
    shuffle=True,
    seed=123,
)



class_names = np.array(ds_train.class_names)
# print(len(class_names))
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


# model=tf.keras.applications.MobileNetV2(
#     input_shape=(500,500,3), alpha=0.35,include_top=False,
#     weights='imagenet',  classes=3
# )
# model.trainable = True
# model.summary()

# image_batch, label_batch = next(iter(ds_train))
# feature_batch = model(image_batch)
# print(feature_batch.shape)
# global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
# feature_batch_average = global_average_layer(feature_batch)
# print(feature_batch_average.shape)
# prediction_layer = tf.keras.layers.Dense(3)
# prediction_batch = prediction_layer(feature_batch_average)
# print(prediction_batch.shape)
# preprocess_input = tf.keras.applications.mobilenet.preprocess_input
# data_augmentation = keras.Sequential(
#     [
#         layers.experimental.preprocessing.RandomFlip("horizontal"),
#         layers.experimental.preprocessing.RandomRotation(0.1),
#     ]
# )

# inputs = tf.keras.Input(shape=(500, 500, 3))
# x = data_augmentation(inputs)
# x = preprocess_input(x)
# x = model(x, training=True)
# x = global_average_layer(x)
# x = layers.Dense(units=1280, activation="relu")(x)
# x = layers.Dropout(0.3)(x)
# # x = tf.keras.layers.Dropout(0.2)(x)
# outputs = prediction_layer(x)
# model = tf.keras.Model(inputs, outputs)

# model.compile(optimizer=tf.keras.optimizers.Adam(0.0001),
#               loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
#               metrics=['accuracy'])
# model.fit(
#     ds_train,
#     epochs=100,
#     validation_data=ds_test,
# )

# model.evaluate(ds_test)

# export_path = "/home/pbenga2s/RnD/Beans_advdef/saved_models/MobileNet/MobileNet_model"
# model.save(export_path)

# export_path


export_path = "/home/pbenga2s/RnD/Beans_advdef/saved_models/MobileNet/MobileNet_model"
model = tf.keras.models.load_model(export_path)

model.evaluate(ds_test)

clean_acc=np.max((tf.nn.softmax(model.predict(ds_test),axis=-1)),axis=-1)

class1_40p_patched_imgs = np.load('/home/pbenga2s/RnD/Beans_advdef/MobileNetV2/patch/class1/patched_images_40p_data.npy')
class1_35p_patched_imgs = np.load('/home/pbenga2s/RnD/Beans_advdef/MobileNetV2/patch/class1/patched_images_35p_data.npy')
class1_25p_patched_imgs = np.load('/home/pbenga2s/RnD/Beans_advdef/MobileNetV2/patch/class1/patched_images_25p_data.npy')
class1_20p_patched_imgs = np.load('/home/pbenga2s/RnD/Beans_advdef/MobileNetV2/patch/class1/patched_images_20p_data.npy')
plt.clf()
pred_labels_40p=np.max(tf.nn.softmax(model.predict(class1_40p_patched_imgs),axis=-1),axis=-1)
pred_labels_35p=np.max(tf.nn.softmax(model.predict(class1_35p_patched_imgs),axis=-1),axis=-1)
pred_labels_25p=np.max(tf.nn.softmax(model.predict(class1_25p_patched_imgs),axis=-1),axis=-1)
pred_labels_20p=np.max(tf.nn.softmax(model.predict(class1_20p_patched_imgs),axis=-1),axis=-1)
data0 = pd.DataFrame({0: clean_acc.tolist()})
data11 = pd.DataFrame({0: pred_labels_35p.tolist()})
data12 = pd.DataFrame({0: pred_labels_25p.tolist()})
data13 = pd.DataFrame({0: pred_labels_20p.tolist()})
data14 = pd.DataFrame({0: pred_labels_40p.tolist()})
data1 = np.hstack((data0,data13))
data1 = np.hstack((data1,data12))
data1 = np.hstack((data1,data11))
data1 = np.hstack((data1,data14))
df = pd.DataFrame(data1,columns=['clean(without patch)','20p', '25p', '35p','40p'])
# a=plt.ylabel('confidence level')
boxplot = df.boxplot(grid=True)
plt.tight_layout()
plt.xlabel('Patch Size')
plt.ylabel('Confidence Level')
plt.title('Model Confidence Plot')
plt.savefig("/home/pbenga2s/RnD/Beans_advdef/MobileNetV2/output/class1.png",bbox_inches='tight')
plt.savefig("/home/pbenga2s/RnD/Beans_advdef/MobileNetV2/output/class1.pdf",bbox_inches='tight')
plt.clf()

class1_40p_patched_imgs = np.load('/home/pbenga2s/RnD/Beans_advdef/MobileNetV2/patch/class2/patched_images_40p_data.npy')
class1_35p_patched_imgs = np.load('/home/pbenga2s/RnD/Beans_advdef/MobileNetV2/patch/class2/patched_images_35p_data.npy')
class1_25p_patched_imgs = np.load('/home/pbenga2s/RnD/Beans_advdef/MobileNetV2/patch/class2/patched_images_25p_data.npy')
class1_20p_patched_imgs = np.load('/home/pbenga2s/RnD/Beans_advdef/MobileNetV2/patch/class2/patched_images_20p_data.npy')
pred_labels_40p=np.max(tf.nn.softmax(model.predict(class1_40p_patched_imgs),axis=-1),axis=-1)
pred_labels_35p=np.max(tf.nn.softmax(model.predict(class1_35p_patched_imgs),axis=-1),axis=-1)
pred_labels_25p=np.max(tf.nn.softmax(model.predict(class1_25p_patched_imgs),axis=-1),axis=-1)
pred_labels_20p=np.max(tf.nn.softmax(model.predict(class1_20p_patched_imgs),axis=-1),axis=-1)
# data0 = pd.DataFrame({0: clean_acc.tolist()})
data21 = pd.DataFrame({0: pred_labels_40p.tolist()})
data22 = pd.DataFrame({0: pred_labels_35p.tolist()})
data23 = pd.DataFrame({0: pred_labels_25p.tolist()})
data24 = pd.DataFrame({0: pred_labels_20p.tolist()})
data2 = np.hstack((data0,data24))
data2 = np.hstack((data2,data23))
data2 = np.hstack((data2,data22))
data2 = np.hstack((data2,data21))
df = pd.DataFrame(data2,columns=['clean(without patch)','20p', '25p', '35p','40p'])
a=plt.ylabel('confidence level')
boxplot = df.boxplot(grid=True)
plt.tight_layout()
plt.xlabel('Patch Size')
plt.ylabel('Confidence Level')
plt.title('Model Confidence Plot')
plt.savefig("/home/pbenga2s/RnD/Beans_advdef/MobileNetV2/output/class2.png",bbox_inches='tight')
plt.savefig("/home/pbenga2s/RnD/Beans_advdef/MobileNetV2/output/class2.pdf",bbox_inches='tight')
plt.clf()

class1_40p_patched_imgs = np.load('/home/pbenga2s/RnD/Beans_advdef/MobileNetV2/patch/class3/patched_images_40p_data.npy')
class1_35p_patched_imgs = np.load('/home/pbenga2s/RnD/Beans_advdef/MobileNetV2/patch/class3/patched_images_35p_data.npy')
class1_25p_patched_imgs = np.load('/home/pbenga2s/RnD/Beans_advdef/MobileNetV2/patch/class3/patched_images_25p_data.npy')
class1_20p_patched_imgs = np.load('/home/pbenga2s/RnD/Beans_advdef/MobileNetV2/patch/class3/patched_images_20p_data.npy')
pred_labels_40p=np.max(tf.nn.softmax(model.predict(class1_40p_patched_imgs),axis=-1),axis=-1)
pred_labels_35p=np.max(tf.nn.softmax(model.predict(class1_35p_patched_imgs),axis=-1),axis=-1)
pred_labels_25p=np.max(tf.nn.softmax(model.predict(class1_25p_patched_imgs),axis=-1),axis=-1)
pred_labels_20p=np.max(tf.nn.softmax(model.predict(class1_20p_patched_imgs),axis=-1),axis=-1)
# data0 = pd.DataFrame({0: clean_acc.tolist()})
data31 = pd.DataFrame({0: pred_labels_40p.tolist()})
data32 = pd.DataFrame({0: pred_labels_35p.tolist()})
data33 = pd.DataFrame({0: pred_labels_25p.tolist()})
data34 = pd.DataFrame({0: pred_labels_20p.tolist()})
data3 = np.hstack((data0,data34))
data3 = np.hstack((data3,data32))
data3 = np.hstack((data3,data33))
data3 = np.hstack((data3,data31))
df = pd.DataFrame(data3,columns=['clean(without patch)','20p', '25p', '35p','40p'])
# a=plt.ylabel('confidence level')
boxplot = df.boxplot(grid=True)
plt.tight_layout()
plt.xlabel('Patch Size')
plt.ylabel('Confidence Level')
plt.title('Model Confidence Plot')
plt.savefig("/home/pbenga2s/RnD/Beans_advdef/MobileNetV2/output/class3.png",bbox_inches='tight')
plt.savefig("/home/pbenga2s/RnD/Beans_advdef/MobileNetV2/output/class3.pdf",bbox_inches='tight')

plt.clf()
data123 = np.hstack((data0,data13))
data123 = np.hstack((data123,data12))
data123 = np.hstack((data123,data11))
data123 = np.hstack((data123,data14))
data123 = np.hstack((data123,data24))
data123 = np.hstack((data123,data23))
data123 = np.hstack((data123,data22))
data123 = np.hstack((data123,data21))
data123 = np.hstack((data123,data34))
data123 = np.hstack((data123,data32))
data123 = np.hstack((data123,data33))
data123 = np.hstack((data123,data31))
df = pd.DataFrame(data123,columns=['clean(without patch)','class1_20p', 'class1_25p', 'class1_35p','class1_40p','class2_20p', 'class2_25p', 'class2_35p','class2_40p','class3_20p', 'class3_25p', 'class3_35p','class3_40p'])
# a=plt.ylabel('confidence level')
boxplot = df.boxplot(grid=True,figsize=(100,10),vert=False)
# plt.tight_layout()
# fig.autofmt_xdate()
plt.xlabel('Patch Size')
plt.ylabel('Confidence Level')
plt.title('Model Confidence Plot')
plt.savefig("/home/pbenga2s/RnD/Beans_advdef/MobileNetV2/output/class123.png",bbox_inches='tight')
plt.savefig("/home/pbenga2s/RnD/Beans_advdef/MobileNetV2/output/class123.pdf",bbox_inches='tight')

plt.clf()
df = pd.DataFrame(data123,columns=['clean(without patch)','class1_20p', 'class1_25p', 'class1_35p','class1_40p','class2_20p', 'class2_25p', 'class2_35p','class2_40p','class3_20p', 'class3_25p', 'class3_35p','class3_40p'])
boxplot = df.boxplot(grid=True,figsize=(100,10))
plt.tight_layout()
plt.gcf().autofmt_xdate()
plt.xlabel('Patch Size')
plt.ylabel('Confidence Level')
plt.title('Model Confidence Plot')
plt.savefig("/home/pbenga2s/RnD/Beans_advdef/MobileNetV2/output/class1233.png",bbox_inches='tight')
plt.savefig("/home/pbenga2s/RnD/Beans_advdef/MobileNetV2/output/class1233.pdf",bbox_inches='tight')











class1_40p_patched_imgs = np.load('/home/pbenga2s/RnD/Beans_advdef/MobileNetV2/patch/class1/patched_images_40p_data.npy')
class1_35p_patched_imgs = np.load('/home/pbenga2s/RnD/Beans_advdef/MobileNetV2/patch/class1/patched_images_35p_data.npy')
class1_25p_patched_imgs = np.load('/home/pbenga2s/RnD/Beans_advdef/MobileNetV2/patch/class1/patched_images_25p_data.npy')
class1_20p_patched_imgs = np.load('/home/pbenga2s/RnD/Beans_advdef/MobileNetV2/patch/class1/patched_images_20p_data.npy')
plt.clf()
pred_labels_40p=np.max(tf.nn.softmax(model.predict(class1_40p_patched_imgs),axis=-1),axis=-1)
pred_labels_35p=np.max(tf.nn.softmax(model.predict(class1_35p_patched_imgs),axis=-1),axis=-1)
pred_labels_25p=np.max(tf.nn.softmax(model.predict(class1_25p_patched_imgs),axis=-1),axis=-1)
pred_labels_20p=np.max(tf.nn.softmax(model.predict(class1_20p_patched_imgs),axis=-1),axis=-1)
data0 = pd.DataFrame({0: clean_acc.tolist()})
data11 = pd.DataFrame({0: pred_labels_35p.tolist()})
data12 = pd.DataFrame({0: pred_labels_25p.tolist()})
data13 = pd.DataFrame({0: pred_labels_20p.tolist()})
data14 = pd.DataFrame({0: pred_labels_40p.tolist()})
data1 = np.hstack((data0,data13))
data1 = np.hstack((data1,data12))
data1 = np.hstack((data1,data11))
data1 = np.hstack((data1,data14))
df = pd.DataFrame(data1,columns=['clean(without patch)','20p', '25p', '35p','40p'])
# a=plt.ylabel('confidence level')
boxplot = df.boxplot(grid=True)
plt.tight_layout()
plt.xlabel('Patch Size')
plt.ylabel('Confidence Level')
plt.title('Model Confidence Plot - Angular Leaf Spot Class')
plt.savefig("/home/pbenga2s/RnD/Beans_advdef/MobileNetV2/output/class1_2.png",bbox_inches='tight')
plt.savefig("/home/pbenga2s/RnD/Beans_advdef/MobileNetV2/output/class1_2.pdf",bbox_inches='tight')
plt.clf()

class1_40p_patched_imgs = np.load('/home/pbenga2s/RnD/Beans_advdef/MobileNetV2/patch/class2/patched_images_40p_data.npy')
class1_35p_patched_imgs = np.load('/home/pbenga2s/RnD/Beans_advdef/MobileNetV2/patch/class2/patched_images_35p_data.npy')
class1_25p_patched_imgs = np.load('/home/pbenga2s/RnD/Beans_advdef/MobileNetV2/patch/class2/patched_images_25p_data.npy')
class1_20p_patched_imgs = np.load('/home/pbenga2s/RnD/Beans_advdef/MobileNetV2/patch/class2/patched_images_20p_data.npy')
pred_labels_40p=np.max(tf.nn.softmax(model.predict(class1_40p_patched_imgs),axis=-1),axis=-1)
pred_labels_35p=np.max(tf.nn.softmax(model.predict(class1_35p_patched_imgs),axis=-1),axis=-1)
pred_labels_25p=np.max(tf.nn.softmax(model.predict(class1_25p_patched_imgs),axis=-1),axis=-1)
pred_labels_20p=np.max(tf.nn.softmax(model.predict(class1_20p_patched_imgs),axis=-1),axis=-1)
# data0 = pd.DataFrame({0: clean_acc.tolist()})
data21 = pd.DataFrame({0: pred_labels_40p.tolist()})
data22 = pd.DataFrame({0: pred_labels_35p.tolist()})
data23 = pd.DataFrame({0: pred_labels_25p.tolist()})
data24 = pd.DataFrame({0: pred_labels_20p.tolist()})
data2 = np.hstack((data0,data24))
data2 = np.hstack((data2,data23))
data2 = np.hstack((data2,data22))
data2 = np.hstack((data2,data21))
df = pd.DataFrame(data2,columns=['clean(without patch)','20p', '25p', '35p','40p'])
a=plt.ylabel('confidence level')
boxplot = df.boxplot(grid=True)
plt.tight_layout()
plt.xlabel('Patch Size')
plt.ylabel('Confidence Level')
plt.title('Model Confidence Plot - Bean Rust Class')
plt.savefig("/home/pbenga2s/RnD/Beans_advdef/MobileNetV2/output/class2_2.png",bbox_inches='tight')
plt.savefig("/home/pbenga2s/RnD/Beans_advdef/MobileNetV2/output/class2_2.pdf",bbox_inches='tight')
plt.clf()

class1_40p_patched_imgs = np.load('/home/pbenga2s/RnD/Beans_advdef/MobileNetV2/patch/class3/patched_images_40p_data.npy')
class1_35p_patched_imgs = np.load('/home/pbenga2s/RnD/Beans_advdef/MobileNetV2/patch/class3/patched_images_35p_data.npy')
class1_25p_patched_imgs = np.load('/home/pbenga2s/RnD/Beans_advdef/MobileNetV2/patch/class3/patched_images_25p_data.npy')
class1_20p_patched_imgs = np.load('/home/pbenga2s/RnD/Beans_advdef/MobileNetV2/patch/class3/patched_images_20p_data.npy')
pred_labels_40p=np.max(tf.nn.softmax(model.predict(class1_40p_patched_imgs),axis=-1),axis=-1)
pred_labels_35p=np.max(tf.nn.softmax(model.predict(class1_35p_patched_imgs),axis=-1),axis=-1)
pred_labels_25p=np.max(tf.nn.softmax(model.predict(class1_25p_patched_imgs),axis=-1),axis=-1)
pred_labels_20p=np.max(tf.nn.softmax(model.predict(class1_20p_patched_imgs),axis=-1),axis=-1)
# data0 = pd.DataFrame({0: clean_acc.tolist()})
data31 = pd.DataFrame({0: pred_labels_40p.tolist()})
data32 = pd.DataFrame({0: pred_labels_35p.tolist()})
data33 = pd.DataFrame({0: pred_labels_25p.tolist()})
data34 = pd.DataFrame({0: pred_labels_20p.tolist()})
data3 = np.hstack((data0,data34))
data3 = np.hstack((data3,data32))
data3 = np.hstack((data3,data33))
data3 = np.hstack((data3,data31))
df = pd.DataFrame(data3,columns=['clean(without patch)','20p', '25p', '35p','40p'])
# a=plt.ylabel('confidence level')
boxplot = df.boxplot(grid=True)
plt.tight_layout()
plt.xlabel('Patch Size')
plt.ylabel('Confidence Level')
plt.title('Model Confidence Plot - Healthy Class')
plt.savefig("/home/pbenga2s/RnD/Beans_advdef/MobileNetV2/output/class3_2.png",bbox_inches='tight')
plt.savefig("/home/pbenga2s/RnD/Beans_advdef/MobileNetV2/output/class3_2.pdf",bbox_inches='tight')