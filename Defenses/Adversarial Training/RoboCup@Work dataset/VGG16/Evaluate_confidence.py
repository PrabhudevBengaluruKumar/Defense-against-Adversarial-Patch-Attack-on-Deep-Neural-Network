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
import os
    





img_height = 64
img_width = 64
batch_size = 32
ds_train = tf.keras.preprocessing.image_dataset_from_directory(
    "/home/pbenga2s/RnD/RoboCup_advdef/VGG16/RoboCup/train",
    labels="inferred",
    label_mode="categorical",  
    color_mode="rgb",
    batch_size=batch_size,
    image_size=(img_height, img_width), 
    shuffle=True,
    seed=123,
)
ds_test = tf.keras.preprocessing.image_dataset_from_directory(
    "/home/pbenga2s/RnD/RoboCup_advdef/VGG16/RoboCup/val",
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


export_path = "/home/pbenga2s/RnD/RoboCup_advdef/saved_models/VGG16/VGG16_model"
model = tf.keras.models.load_model(export_path)
model.evaluate(ds_test)



patched_images_40p_class12_data_path = '/home/pbenga2s/RnD/RoboCup_advdef/VGG16/patch/class12/patched_images_40p_data.npy'
patched_images_35p_class12_data_path = '/home/pbenga2s/RnD/RoboCup_advdef/VGG16/patch/class12/patched_images_35p_data.npy'
patched_images_25p_class12_data_path = '/home/pbenga2s/RnD/RoboCup_advdef/VGG16/patch/class12/patched_images_25p_data.npy'
patched_images_20p_class12_data_path = '/home/pbenga2s/RnD/RoboCup_advdef/VGG16/patch/class12/patched_images_20p_data.npy'

patched_images_40p_class3_data_path = '/home/pbenga2s/RnD/RoboCup_advdef/VGG16/patch/class3/patched_images_40p_data.npy'
patched_images_35p_class3_data_path = '/home/pbenga2s/RnD/RoboCup_advdef/VGG16/patch/class3/patched_images_35p_data.npy'
patched_images_25p_class3_data_path = '/home/pbenga2s/RnD/RoboCup_advdef/VGG16/patch/class3/patched_images_25p_data.npy'
patched_images_20p_class3_data_path = '/home/pbenga2s/RnD/RoboCup_advdef/VGG16/patch/class3/patched_images_20p_data.npy'

patched_images_40p_class6_data_path = '/home/pbenga2s/RnD/RoboCup_advdef/VGG16/patch/class6/patched_images_40p_data.npy'
patched_images_35p_class6_data_path = '/home/pbenga2s/RnD/RoboCup_advdef/VGG16/patch/class6/patched_images_35p_data.npy'
patched_images_25p_class6_data_path = '/home/pbenga2s/RnD/RoboCup_advdef/VGG16/patch/class6/patched_images_25p_data.npy'
patched_images_20p_class6_data_path = '/home/pbenga2s/RnD/RoboCup_advdef/VGG16/patch/class6/patched_images_20p_data.npy'

patched_images_40p_class7_data_path = '/home/pbenga2s/RnD/RoboCup_advdef/VGG16/patch/class7/patched_images_40p_data.npy'
patched_images_35p_class7_data_path = '/home/pbenga2s/RnD/RoboCup_advdef/VGG16/patch/class7/patched_images_35p_data.npy'
patched_images_25p_class7_data_path = '/home/pbenga2s/RnD/RoboCup_advdef/VGG16/patch/class7/patched_images_25p_data.npy'
patched_images_20p_class7_data_path = '/home/pbenga2s/RnD/RoboCup_advdef/VGG16/patch/class7/patched_images_20p_data.npy'

patched_images_40p_class9_data_path = '/home/pbenga2s/RnD/RoboCup_advdef/VGG16/patch/class9/patched_images_40p_data.npy'
patched_images_35p_class9_data_path = '/home/pbenga2s/RnD/RoboCup_advdef/VGG16/patch/class9/patched_images_35p_data.npy'
patched_images_25p_class9_data_path = '/home/pbenga2s/RnD/RoboCup_advdef/VGG16/patch/class9/patched_images_25p_data.npy'
patched_images_20p_class9_data_path = '/home/pbenga2s/RnD/RoboCup_advdef/VGG16/patch/class9/patched_images_20p_data.npy'

# export_path_save_fig = os.path.abspath('/home/pbenga2s/RnD/RoboCup_advdef/MobileNet/output/')



clean_acc=np.max((tf.nn.softmax(model.predict(ds_test),axis=-1)),axis=-1)



class12_40p_patched_imgs = np.load(patched_images_40p_class12_data_path)
class12_35p_patched_imgs = np.load(patched_images_35p_class12_data_path)
class12_25p_patched_imgs = np.load(patched_images_25p_class12_data_path)
class12_20p_patched_imgs = np.load(patched_images_20p_class12_data_path)
plt.clf()
pred_labels_40p_class12=np.max(tf.nn.softmax(model.predict(class12_40p_patched_imgs),axis=-1),axis=-1)
pred_labels_35p_class12=np.max(tf.nn.softmax(model.predict(class12_35p_patched_imgs),axis=-1),axis=-1)
pred_labels_25p_class12=np.max(tf.nn.softmax(model.predict(class12_25p_patched_imgs),axis=-1),axis=-1)
pred_labels_20p_class12=np.max(tf.nn.softmax(model.predict(class12_20p_patched_imgs),axis=-1),axis=-1)
data0 = pd.DataFrame({0: clean_acc.tolist()})
data122 = pd.DataFrame({0: pred_labels_35p_class12.tolist()})
data123 = pd.DataFrame({0: pred_labels_25p_class12.tolist()})
data124 = pd.DataFrame({0: pred_labels_20p_class12.tolist()})
data121 = pd.DataFrame({0: pred_labels_40p_class12.tolist()})
data12 = np.hstack((data0,data124))
data12 = np.hstack((data12,data123))
data12 = np.hstack((data12,data122))
data12 = np.hstack((data12,data121))
df = pd.DataFrame(data12,columns=['clean(without patch)','20p', '25p', '35p','40p'])
boxplot = df.boxplot(grid=True)
plt.tight_layout()
plt.xlabel('Patch Size')
plt.ylabel('Confidence Level')
plt.title('Model Confidence Level Plot')
plt.savefig('/home/pbenga2s/RnD/RoboCup_advdef/VGG16/output/class12.png',bbox_inches='tight')
plt.savefig("/home/pbenga2s/RnD/RoboCup_advdef/VGG16/output/class12.pdf",bbox_inches='tight')
plt.clf()




class3_40p_patched_imgs = np.load(patched_images_40p_class3_data_path)
class3_35p_patched_imgs = np.load(patched_images_35p_class3_data_path)
class3_25p_patched_imgs = np.load(patched_images_25p_class3_data_path)
class3_20p_patched_imgs = np.load(patched_images_20p_class3_data_path)
pred_labels_40p_class3=np.max(tf.nn.softmax(model.predict(class3_40p_patched_imgs),axis=-1),axis=-1)
pred_labels_35p_class3=np.max(tf.nn.softmax(model.predict(class3_35p_patched_imgs),axis=-1),axis=-1)
pred_labels_25p_class3=np.max(tf.nn.softmax(model.predict(class3_25p_patched_imgs),axis=-1),axis=-1)
pred_labels_20p_class3=np.max(tf.nn.softmax(model.predict(class3_20p_patched_imgs),axis=-1),axis=-1)
# data0 = pd.DataFrame({0: clean_acc.tolist()})
data31 = pd.DataFrame({0: pred_labels_40p_class3.tolist()})
data32 = pd.DataFrame({0: pred_labels_35p_class3.tolist()})
data33 = pd.DataFrame({0: pred_labels_25p_class3.tolist()})
data34 = pd.DataFrame({0: pred_labels_20p_class3.tolist()})
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
plt.title('Model Confidence Level Plot')
plt.savefig("/home/pbenga2s/RnD/RoboCup_advdef/VGG16/output/class3.png",bbox_inches='tight')
plt.savefig("/home/pbenga2s/RnD/RoboCup_advdef/VGG16/output/class3.pdf",bbox_inches='tight')



class6_40p_patched_imgs = np.load(patched_images_40p_class6_data_path)
class6_35p_patched_imgs = np.load(patched_images_35p_class6_data_path)
class6_25p_patched_imgs = np.load(patched_images_25p_class6_data_path)
class6_20p_patched_imgs = np.load(patched_images_20p_class6_data_path)
pred_labels_40p_class6 = np.max(tf.nn.softmax(model.predict(class6_40p_patched_imgs),axis=-1),axis=-1)
pred_labels_35p_class6 = np.max(tf.nn.softmax(model.predict(class6_35p_patched_imgs),axis=-1),axis=-1)
pred_labels_25p_class6 = np.max(tf.nn.softmax(model.predict(class6_25p_patched_imgs),axis=-1),axis=-1)
pred_labels_20p_class6 = np.max(tf.nn.softmax(model.predict(class6_20p_patched_imgs),axis=-1),axis=-1)
data61 = pd.DataFrame({0: pred_labels_40p_class6.tolist()})
data62 = pd.DataFrame({0: pred_labels_35p_class6.tolist()})
data63 = pd.DataFrame({0: pred_labels_25p_class6.tolist()})
data64 = pd.DataFrame({0: pred_labels_20p_class6.tolist()})
data6 = np.hstack((data0,data64))
data6 = np.hstack((data6,data63))
data6 = np.hstack((data6,data62))
data6 = np.hstack((data6,data61))
df = pd.DataFrame(data6,columns=['clean(without patch)','20p', '25p', '35p','40p'])
a=plt.ylabel('confidence level')
boxplot = df.boxplot(grid=True)
plt.tight_layout()
plt.xlabel('Patch Size')
plt.ylabel('Confidence Level')
plt.title('Model Confidence Level Plot')
plt.savefig("/home/pbenga2s/RnD/RoboCup_advdef/VGG16/output/class6.png",bbox_inches='tight')
plt.savefig("/home/pbenga2s/RnD/RoboCup_advdef/VGG16/output/class6.pdf",bbox_inches='tight')
plt.clf()




class7_40p_patched_imgs = np.load(patched_images_40p_class7_data_path)
class7_35p_patched_imgs = np.load(patched_images_35p_class7_data_path)
class7_25p_patched_imgs = np.load(patched_images_25p_class7_data_path)
class7_20p_patched_imgs = np.load(patched_images_20p_class7_data_path)
pred_labels_40p_class7 = np.max(tf.nn.softmax(model.predict(class7_40p_patched_imgs),axis=-1),axis=-1)
pred_labels_35p_class7 = np.max(tf.nn.softmax(model.predict(class7_35p_patched_imgs),axis=-1),axis=-1)
pred_labels_25p_class7 = np.max(tf.nn.softmax(model.predict(class7_25p_patched_imgs),axis=-1),axis=-1)
pred_labels_20p_class7 = np.max(tf.nn.softmax(model.predict(class7_20p_patched_imgs),axis=-1),axis=-1)
data71 = pd.DataFrame({0: pred_labels_40p_class7.tolist()})
data72 = pd.DataFrame({0: pred_labels_35p_class7.tolist()})
data73 = pd.DataFrame({0: pred_labels_25p_class7.tolist()})
data74 = pd.DataFrame({0: pred_labels_20p_class7.tolist()})
data7 = np.hstack((data0,data64))
data7 = np.hstack((data7,data63))
data7 = np.hstack((data7,data62))
data7 = np.hstack((data7,data61))
df = pd.DataFrame(data7,columns=['clean(without patch)','20p', '25p', '35p','40p'])
a=plt.ylabel('confidence level')
boxplot = df.boxplot(grid=True)
plt.tight_layout()
plt.xlabel('Patch Size')
plt.ylabel('Confidence Level')
plt.title('Model Confidence Level Plot')
plt.savefig("/home/pbenga2s/RnD/RoboCup_advdef/VGG16/output/class7.png",bbox_inches='tight')
plt.savefig("/home/pbenga2s/RnD/RoboCup_advdef/VGG16/output/class7.pdf",bbox_inches='tight')
plt.clf()




class9_40p_patched_imgs = np.load(patched_images_40p_class9_data_path)
class9_35p_patched_imgs = np.load(patched_images_35p_class9_data_path)
class9_25p_patched_imgs = np.load(patched_images_25p_class9_data_path)
class9_20p_patched_imgs = np.load(patched_images_20p_class9_data_path)
pred_labels_40p_class9 = np.max(tf.nn.softmax(model.predict(class9_40p_patched_imgs),axis=-1),axis=-1)
pred_labels_35p_class9 = np.max(tf.nn.softmax(model.predict(class9_35p_patched_imgs),axis=-1),axis=-1)
pred_labels_25p_class9 = np.max(tf.nn.softmax(model.predict(class9_25p_patched_imgs),axis=-1),axis=-1)
pred_labels_20p_class9 = np.max(tf.nn.softmax(model.predict(class9_20p_patched_imgs),axis=-1),axis=-1)
data91 = pd.DataFrame({0: pred_labels_40p_class9.tolist()})
data92 = pd.DataFrame({0: pred_labels_35p_class9.tolist()})
data93 = pd.DataFrame({0: pred_labels_25p_class9.tolist()})
data94 = pd.DataFrame({0: pred_labels_20p_class9.tolist()})
data9 = np.hstack((data0,data94))
data9 = np.hstack((data9,data93))
data9 = np.hstack((data9,data92))
data9 = np.hstack((data9,data91))
df = pd.DataFrame(data9,columns=['clean(without patch)','20p', '25p', '35p','40p'])
a=plt.ylabel('confidence level')
boxplot = df.boxplot(grid=True)
plt.tight_layout()
plt.xlabel('Patch Size')
plt.ylabel('Confidence Level')
plt.title('Model Confidence Level Plot')
plt.savefig("/home/pbenga2s/RnD/RoboCup_advdef/VGG16/output/class9.png",bbox_inches='tight')
plt.savefig("/home/pbenga2s/RnD/RoboCup_advdef/VGG16/output/class9.pdf",bbox_inches='tight')
plt.clf()




plt.clf()

data13679 = np.hstack((data0,data34))
data13679 = np.hstack((data13679,data33))
data13679 = np.hstack((data13679,data32))
data13679 = np.hstack((data13679,data31))
data13679 = np.hstack((data13679,data64))
data13679 = np.hstack((data13679,data62))
data13679 = np.hstack((data13679,data63))
data13679 = np.hstack((data13679,data61))
data13679 = np.hstack((data13679,data74))
data13679 = np.hstack((data13679,data72))
data13679 = np.hstack((data13679,data73))
data13679 = np.hstack((data13679,data71))
data13679 = np.hstack((data13679,data94))
data13679 = np.hstack((data13679,data92))
data13679 = np.hstack((data13679,data93))
data13679 = np.hstack((data13679,data91))
data13679 = np.hstack((data13679,data124))
data13679 = np.hstack((data13679,data123))
data13679 = np.hstack((data13679,data122))
data13679 = np.hstack((data13679,data121))
df = pd.DataFrame(data13679,columns=['clean(without patch)','class3_20p', 'class3_25p', 'class3_35p','class3_40p','class6_20p', 'class6_25p', 'class6_35p','class6_40p','class7_20p', 'class7_25p', 'class7_35p','class7_40p','class9_20p', 'class9_25p', 'class9_35p','class9_40p','class12_20p', 'class12_25p', 'class12_35p','class12_40p'])
# a=plt.ylabel('confidence level')
boxplot = df.boxplot(grid=True,figsize=(100,10),vert=False)
# plt.tight_layout()
# fig.autofmt_xdate()
plt.xlabel('Patch Size')
plt.ylabel('Confidence Level')
plt.title('Model Confidence Level Plot')
plt.savefig("/home/pbenga2s/RnD/RoboCup_advdef/VGG16/output/class13679.png",bbox_inches='tight')
plt.savefig("/home/pbenga2s/RnD/RoboCup_advdef/VGG16/output/class13679.pdf",bbox_inches='tight')




plt.clf()
df = pd.DataFrame(data13679,columns=['clean(without patch)','class3_20p', 'class3_25p', 'class3_35p','class3_40p','class6_20p', 'class6_25p', 'class6_35p','class6_40p','class7_20p', 'class7_25p', 'class7_35p','class7_40p','class9_20p', 'class9_25p', 'class9_35p','class9_40p','class12_20p', 'class12_25p', 'class12_35p','class12_40p'])
boxplot = df.boxplot(grid=True,figsize=(100,10))
plt.tight_layout()
plt.gcf().autofmt_xdate()
plt.xlabel('Patch Size')
plt.ylabel('Confidence Level')
plt.title('Model Confidence Level Plot')
plt.savefig("/home/pbenga2s/RnD/RoboCup_advdef/VGG16/output/class13679_1.png",bbox_inches='tight')
plt.savefig("/home/pbenga2s/RnD/RoboCup_advdef/VGG16/output/class13679_1.pdf",bbox_inches='tight')









class12_40p_patched_imgs = np.load(patched_images_40p_class12_data_path)
class12_35p_patched_imgs = np.load(patched_images_35p_class12_data_path)
class12_25p_patched_imgs = np.load(patched_images_25p_class12_data_path)
class12_20p_patched_imgs = np.load(patched_images_20p_class12_data_path)
plt.clf()
pred_labels_40p_class12=np.max(tf.nn.softmax(model.predict(class12_40p_patched_imgs),axis=-1),axis=-1)
pred_labels_35p_class12=np.max(tf.nn.softmax(model.predict(class12_35p_patched_imgs),axis=-1),axis=-1)
pred_labels_25p_class12=np.max(tf.nn.softmax(model.predict(class12_25p_patched_imgs),axis=-1),axis=-1)
pred_labels_20p_class12=np.max(tf.nn.softmax(model.predict(class12_20p_patched_imgs),axis=-1),axis=-1)
data0 = pd.DataFrame({0: clean_acc.tolist()})
data122 = pd.DataFrame({0: pred_labels_35p_class12.tolist()})
data123 = pd.DataFrame({0: pred_labels_25p_class12.tolist()})
data124 = pd.DataFrame({0: pred_labels_20p_class12.tolist()})
data121 = pd.DataFrame({0: pred_labels_40p_class12.tolist()})
data12 = np.hstack((data0,data124))
data12 = np.hstack((data12,data123))
data12 = np.hstack((data12,data122))
data12 = np.hstack((data12,data121))
df = pd.DataFrame(data12,columns=['clean(without patch)','20p', '25p', '35p','40p'])
boxplot = df.boxplot(grid=True)
plt.tight_layout()
plt.xlabel('Patch Size')
plt.ylabel('Confidence Level')
plt.title('Model Confidence Level Plot - Motor Class')
plt.savefig('/home/pbenga2s/RnD/RoboCup_advdef/VGG16/output/class12_2.png',bbox_inches='tight')
plt.savefig("/home/pbenga2s/RnD/RoboCup_advdef/VGG16/output/class12_2.pdf",bbox_inches='tight')
plt.clf()




class3_40p_patched_imgs = np.load(patched_images_40p_class3_data_path)
class3_35p_patched_imgs = np.load(patched_images_35p_class3_data_path)
class3_25p_patched_imgs = np.load(patched_images_25p_class3_data_path)
class3_20p_patched_imgs = np.load(patched_images_20p_class3_data_path)
pred_labels_40p_class3=np.max(tf.nn.softmax(model.predict(class3_40p_patched_imgs),axis=-1),axis=-1)
pred_labels_35p_class3=np.max(tf.nn.softmax(model.predict(class3_35p_patched_imgs),axis=-1),axis=-1)
pred_labels_25p_class3=np.max(tf.nn.softmax(model.predict(class3_25p_patched_imgs),axis=-1),axis=-1)
pred_labels_20p_class3=np.max(tf.nn.softmax(model.predict(class3_20p_patched_imgs),axis=-1),axis=-1)
# data0 = pd.DataFrame({0: clean_acc.tolist()})
data31 = pd.DataFrame({0: pred_labels_40p_class3.tolist()})
data32 = pd.DataFrame({0: pred_labels_35p_class3.tolist()})
data33 = pd.DataFrame({0: pred_labels_25p_class3.tolist()})
data34 = pd.DataFrame({0: pred_labels_20p_class3.tolist()})
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
plt.title('Model Confidence Level Plot - Bearing Box Class')
plt.savefig("/home/pbenga2s/RnD/RoboCup_advdef/VGG16/output/class3_2.png",bbox_inches='tight')
plt.savefig("/home/pbenga2s/RnD/RoboCup_advdef/VGG16/output/class3_2.pdf",bbox_inches='tight')



class6_40p_patched_imgs = np.load(patched_images_40p_class6_data_path)
class6_35p_patched_imgs = np.load(patched_images_35p_class6_data_path)
class6_25p_patched_imgs = np.load(patched_images_25p_class6_data_path)
class6_20p_patched_imgs = np.load(patched_images_20p_class6_data_path)
pred_labels_40p_class6 = np.max(tf.nn.softmax(model.predict(class6_40p_patched_imgs),axis=-1),axis=-1)
pred_labels_35p_class6 = np.max(tf.nn.softmax(model.predict(class6_35p_patched_imgs),axis=-1),axis=-1)
pred_labels_25p_class6 = np.max(tf.nn.softmax(model.predict(class6_25p_patched_imgs),axis=-1),axis=-1)
pred_labels_20p_class6 = np.max(tf.nn.softmax(model.predict(class6_20p_patched_imgs),axis=-1),axis=-1)
data61 = pd.DataFrame({0: pred_labels_40p_class6.tolist()})
data62 = pd.DataFrame({0: pred_labels_35p_class6.tolist()})
data63 = pd.DataFrame({0: pred_labels_25p_class6.tolist()})
data64 = pd.DataFrame({0: pred_labels_20p_class6.tolist()})
data6 = np.hstack((data0,data64))
data6 = np.hstack((data6,data63))
data6 = np.hstack((data6,data62))
data6 = np.hstack((data6,data61))
df = pd.DataFrame(data6,columns=['clean(without patch)','20p', '25p', '35p','40p'])
a=plt.ylabel('confidence level')
boxplot = df.boxplot(grid=True)
plt.tight_layout()
plt.xlabel('Patch Size')
plt.ylabel('Confidence Level')
plt.title('Model Confidence Level Plot - Distance Tube Class')
plt.savefig("/home/pbenga2s/RnD/RoboCup_advdef/VGG16/output/class6_2.png",bbox_inches='tight')
plt.savefig("/home/pbenga2s/RnD/RoboCup_advdef/VGG16/output/class6_2.pdf",bbox_inches='tight')
plt.clf()




class7_40p_patched_imgs = np.load(patched_images_40p_class7_data_path)
class7_35p_patched_imgs = np.load(patched_images_35p_class7_data_path)
class7_25p_patched_imgs = np.load(patched_images_25p_class7_data_path)
class7_20p_patched_imgs = np.load(patched_images_20p_class7_data_path)
pred_labels_40p_class7 = np.max(tf.nn.softmax(model.predict(class7_40p_patched_imgs),axis=-1),axis=-1)
pred_labels_35p_class7 = np.max(tf.nn.softmax(model.predict(class7_35p_patched_imgs),axis=-1),axis=-1)
pred_labels_25p_class7 = np.max(tf.nn.softmax(model.predict(class7_25p_patched_imgs),axis=-1),axis=-1)
pred_labels_20p_class7 = np.max(tf.nn.softmax(model.predict(class7_20p_patched_imgs),axis=-1),axis=-1)
data71 = pd.DataFrame({0: pred_labels_40p_class7.tolist()})
data72 = pd.DataFrame({0: pred_labels_35p_class7.tolist()})
data73 = pd.DataFrame({0: pred_labels_25p_class7.tolist()})
data74 = pd.DataFrame({0: pred_labels_20p_class7.tolist()})
data7 = np.hstack((data0,data64))
data7 = np.hstack((data7,data63))
data7 = np.hstack((data7,data62))
data7 = np.hstack((data7,data61))
df = pd.DataFrame(data7,columns=['clean(without patch)','20p', '25p', '35p','40p'])
a=plt.ylabel('confidence level')
boxplot = df.boxplot(grid=True)
plt.tight_layout()
plt.xlabel('Patch Size')
plt.ylabel('Confidence Level')
plt.title('Model Confidence Level Plot - F20_20_B Class')
plt.savefig("/home/pbenga2s/RnD/RoboCup_advdef/VGG16/output/class7_2.png",bbox_inches='tight')
plt.savefig("/home/pbenga2s/RnD/RoboCup_advdef/VGG16/output/class7_2.pdf",bbox_inches='tight')
plt.clf()




class9_40p_patched_imgs = np.load(patched_images_40p_class9_data_path)
class9_35p_patched_imgs = np.load(patched_images_35p_class9_data_path)
class9_25p_patched_imgs = np.load(patched_images_25p_class9_data_path)
class9_20p_patched_imgs = np.load(patched_images_20p_class9_data_path)
pred_labels_40p_class9 = np.max(tf.nn.softmax(model.predict(class9_40p_patched_imgs),axis=-1),axis=-1)
pred_labels_35p_class9 = np.max(tf.nn.softmax(model.predict(class9_35p_patched_imgs),axis=-1),axis=-1)
pred_labels_25p_class9 = np.max(tf.nn.softmax(model.predict(class9_25p_patched_imgs),axis=-1),axis=-1)
pred_labels_20p_class9 = np.max(tf.nn.softmax(model.predict(class9_20p_patched_imgs),axis=-1),axis=-1)
data91 = pd.DataFrame({0: pred_labels_40p_class9.tolist()})
data92 = pd.DataFrame({0: pred_labels_35p_class9.tolist()})
data93 = pd.DataFrame({0: pred_labels_25p_class9.tolist()})
data94 = pd.DataFrame({0: pred_labels_20p_class9.tolist()})
data9 = np.hstack((data0,data94))
data9 = np.hstack((data9,data93))
data9 = np.hstack((data9,data92))
data9 = np.hstack((data9,data91))
df = pd.DataFrame(data9,columns=['clean(without patch)','20p', '25p', '35p','40p'])
a=plt.ylabel('confidence level')
boxplot = df.boxplot(grid=True)
plt.tight_layout()
plt.xlabel('Patch Size')
plt.ylabel('Confidence Level')
plt.title('Model Confidence Level Plot - M20 Class')
plt.savefig("/home/pbenga2s/RnD/RoboCup_advdef/VGG16/output/class9_2.png",bbox_inches='tight')
plt.savefig("/home/pbenga2s/RnD/RoboCup_advdef/VGG16/output/class9_2.pdf",bbox_inches='tight')
plt.clf()




plt.clf()
data13679 = np.hstack((data0,data34))
data13679 = np.hstack((data13679,data33))
data13679 = np.hstack((data13679,data32))
data13679 = np.hstack((data13679,data31))
data13679 = np.hstack((data13679,data64))
data13679 = np.hstack((data13679,data62))
data13679 = np.hstack((data13679,data63))
data13679 = np.hstack((data13679,data61))
data13679 = np.hstack((data13679,data74))
data13679 = np.hstack((data13679,data72))
data13679 = np.hstack((data13679,data73))
data13679 = np.hstack((data13679,data71))
data13679 = np.hstack((data13679,data94))
data13679 = np.hstack((data13679,data92))
data13679 = np.hstack((data13679,data93))
data13679 = np.hstack((data13679,data91))
data13679 = np.hstack((data13679,data124))
data13679 = np.hstack((data13679,data123))
data13679 = np.hstack((data13679,data122))
data13679 = np.hstack((data13679,data121))
df = pd.DataFrame(data13679,columns=['clean(without patch)','Bearing_Box_20p', 'Bearing_Box_25p', 'Bearing_Box_35p','Bearing_Box_40p','Distance_Tube_20p', 'Distance_Tube_25p', 'Distance_Tube_35p','Distance_Tube_40p','F20_20_B_20p', 'F20_20_B_25p', 'F20_20_B_35p','F20_20_B_40p','M20_20p', 'M20_25p', 'M20_35p','M20_40p','Motor_20p', 'Motor_25p', 'Motor_35p','Motor_40p'])
# a=plt.ylabel('confidence level')
boxplot = df.boxplot(grid=True,figsize=(100,10),vert=False)
# plt.tight_layout()
# fig.autofmt_xdate()
plt.xlabel('Patch Size')
plt.ylabel('Confidence Level')
plt.title('Model Confidence Level Plot')
plt.savefig("/home/pbenga2s/RnD/RoboCup_advdef/VGG16/output/class13679_2.png",bbox_inches='tight')
plt.savefig("/home/pbenga2s/RnD/RoboCup_advdef/VGG16/output/class13679_2.pdf",bbox_inches='tight')




plt.clf()
df = pd.DataFrame(data13679,columns=['clean(without patch)','Bearing_Box_20p', 'Bearing_Box_25p', 'Bearing_Box_35p','Bearing_Box_40p','Distance_Tube_20p', 'Distance_Tube_25p', 'Distance_Tube_35p','Distance_Tube_40p','F20_20_B_20p', 'F20_20_B_25p', 'F20_20_B_35p','F20_20_B_40p','M20_20p', 'M20_25p', 'M20_35p','M20_40p','Motor_20p', 'Motor_25p', 'Motor_35p','Motor_40p'])
boxplot = df.boxplot(grid=True,figsize=(100,10))
plt.tight_layout()
plt.gcf().autofmt_xdate()
plt.xlabel('Patch Size')
plt.ylabel('Confidence Level')
plt.title('Model Confidence Level Plot')
plt.savefig("/home/pbenga2s/RnD/RoboCup_advdef/VGG16/output/class13679_22.png",bbox_inches='tight')
plt.savefig("/home/pbenga2s/RnD/RoboCup_advdef/VGG16/output/class13679_22.pdf",bbox_inches='tight')