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

print("MobileNet - Imagenette_UE")
print("class 9")


# img_height = 160
# img_width = 160
# batch_size = 32
# ds_train = tf.keras.preprocessing.image_dataset_from_directory(
#     "/home/pbenga2s/RnD/Imagenette/Imagenette/train",
#     labels="inferred",
#     label_mode="categorical",  
#     color_mode="rgb",
#     batch_size=batch_size,
#     image_size=(img_height, img_width), 
#     shuffle=True,
#     seed=123,
# )
# ds_test = tf.keras.preprocessing.image_dataset_from_directory(
#     "/home/pbenga2s/RnD/Imagenette/Imagenette/val",
#     labels="inferred",
#     label_mode="categorical",
#     color_mode="rgb",
#     batch_size=batch_size,
#     image_size=(img_height, img_width),
#     shuffle=True,
#     seed=123,
# )



# class_names = np.array(ds_train.class_names)
# print(len(class_names))
# normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
# ds_train = ds_train.map(lambda x, y: (normalization_layer(x), y))
# AUTOTUNE = tf.data.AUTOTUNE
# ds_train = ds_train.shuffle(42,reshuffle_each_iteration=False)
# ds_train = ds_train.cache().prefetch(buffer_size=AUTOTUNE)
# normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
# ds_test = ds_test.map(lambda x, y: (normalization_layer(x), y))
# AUTOTUNE = tf.data.AUTOTUNE
# ds_test = ds_test.shuffle(42,reshuffle_each_iteration=False)
# ds_test = ds_test.cache().prefetch(buffer_size=AUTOTUNE)



# export_path = "/home/pbenga2s/RnD/Imagenette/saved_models/MobileNet/MobileNet_model(0.0001)"
# model = tf.keras.models.load_model(export_path)

# model.evaluate(ds_test)



(ds_train,ds_test),ds_info = tfds.load(
    'imagenette/160px', 
    split=['train','validation'], 
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
    )
label_names = ds_info.features['label'].names
print (label_names)
class_names=['fish','dog','radio','chainsaw','building','mellophone','truck','pump_station','ball','parachute']
print(class_names)

def one_hot(image, label):
    """
    Converts the label to categorical.
    Arguments ~
        image: Tensor of Shape (224,224,3) - Simply for outputting
        label: Tensor of Shape (32,) for casting and converting to categorical
    Returns the image (as it was inputted) and the label converted to a categorical vector
    """
    # Casts to an Int and performs one-hot ops
    label = tf.one_hot(tf.cast(label, tf.int32), 10)
    # Recasts it to Float32
    label = tf.cast(label, tf.float32)
    return image, label

def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  image = tf.image.resize(image,[160,160])
  image = tf.cast(image, tf.float32) / 255.
#   return tf.cast(image, tf.float32) / 255., label
  return image,label
ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
# Let's one-hot encode the labels
ds_train = ds_train.map(one_hot)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(32)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_test = ds_test.map(one_hot)
ds_test = ds_test.batch(32)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

export_path = "/home/pbenga2s/RnD/Imagenette_UE/saved_models/MobileNet/MobileNetV2_model(1)"
model = tf.keras.models.load_model(export_path,compile=False)



pred_labels=np.argmax(model.predict(ds_test),axis=-1)
val_list=[]
for a,b in ds_test:
  b=np.argmax(b, axis=1)
  val_list.append(b)
val_list1=[]
for i in val_list:
  for j in i:
    val_list1.append(j)
y=val_list1
cm=tf.math.confusion_matrix(
    y,
    pred_labels,
    
    num_classes=10
)
print("Accuracy clean : ",accuracy_score(y, pred_labels))
sns_plot = sns.heatmap(cm,xticklabels=class_names,yticklabels=class_names, annot=True, fmt='g')
# sns_plot.figure.savefig("/home/pbenga2s/RnD/Imagenette_OOD/MobileNet/patch/class1/cf_35p.png")
plt.tight_layout()
plt.xlabel('Prediction')
plt.ylabel('Label')
plt.title('Confusion Matrix')
plt.savefig("/home/pbenga2s/RnD/Imagenette_UE/MobileNet/patch/class9/cf_clean.png",bbox_inches='tight')
plt.savefig("/home/pbenga2s/RnD/Imagenette_UE/MobileNet/patch/class9/cf_clean.pdf",bbox_inches='tight')

image_shape = (160, 160, 3)
clip_values = (0, 255)
nb_classes  =10
batch_size = 16
scale_min = 0.4
scale_max = 1.0
rotation_max = 22.5
learning_rate = 5000.
max_iter = 500

def bgr_to_rgb(x):
    return x[:, :, ::-1]
def decode_prediction_custom(preds,top=5):
  results = []
  for pred in preds:
    top_indices = pred.argsort()[-top:][::-1]
    result = [(label_names[i],) + (pred[i],) for i in top_indices]
    result.sort(key=lambda x: x[1], reverse=True)
    results.append(result)
  return results
def predict_model(classifier, image):
    plt.imshow(bgr_to_rgb(image))
    plt.show()
    image = np.copy(image)
    image = np.expand_dims(image, axis=0)
    prediction = tf.nn.softmax(classifier.predict(image)).numpy()
    top = 2
    prediction_decode = decode_prediction_custom(prediction, top=top)[0]
    print('Predictions:')
    lengths = list()
    for i in range(top):
        lengths.append(len(prediction_decode[i][0]))
    max_length = max(lengths)
    for i in range(top):
        name = prediction_decode[i][0]
        probability = prediction_decode[i][1]
        output_str = "{} {:.2f}".format(name, probability)
        print(output_str)

images_list = list()
for ims, labels in ds_test:
    for im in ims:
      im = np.array(im)[:, :, ::-1].astype(np.float32) # RGB to BGR
      im = np.expand_dims(im, axis=0)
      images_list.append(im)
images = np.vstack(images_list)

tfc = TensorFlowV2Classifier(model=model, loss_object=None, train_step=None, nb_classes=nb_classes,
                             input_shape=image_shape, clip_values=clip_values)
ap = AdversarialPatch(classifier=tfc, rotation_max=rotation_max, scale_min=scale_min, scale_max=scale_max,
                      learning_rate=learning_rate, max_iter=max_iter, batch_size=batch_size,
                      patch_shape=image_shape)
y_one_hot = np.zeros(nb_classes)
y_one_hot[8] = 1.0
y_target = np.tile(y_one_hot, (images.shape[0], 1))
patch, patch_mask = ap.generate(x=images[:39,:,:], y=y_target[:39])

plt.imsave('/home/pbenga2s/RnD/Imagenette_UE/MobileNet/patch/class9/patch.png',(bgr_to_rgb(patch) * patch_mask).astype(np.uint8))

patched_images_40p = ap.apply_patch(images, scale=0.40)
patched_images_35p = ap.apply_patch(images, scale=0.35)
patched_images_25p = ap.apply_patch(images, scale=0.25)
patched_images_20p = ap.apply_patch(images, scale=0.20)

np.save('/home/pbenga2s/RnD/Imagenette_UE/MobileNet/patch/class9/patch.npy', patch)
np.save('/home/pbenga2s/RnD/Imagenette_UE/MobileNet/patch/class9/patched_images_40p_data.npy', patched_images_40p)
np.save('/home/pbenga2s/RnD/Imagenette_UE/MobileNet/patch/class9/patched_images_35p_data.npy', patched_images_35p)
np.save('/home/pbenga2s/RnD/Imagenette_UE/MobileNet/patch/class9/patched_images_25p_data.npy', patched_images_25p)
np.save('/home/pbenga2s/RnD/Imagenette_UE/MobileNet/patch/class9/patched_images_20p_data.npy', patched_images_20p)

pred_labels_40p=np.argmax(model.predict(patched_images_40p),axis=-1)
# y = np.concatenate([y for x,y in validation_ds],axis=0)
val_list=[]
for a,b in ds_test:
  b=np.argmax(b, axis=1)
  val_list.append(b)
val_list1=[]
for i in val_list:
  for j in i:
    val_list1.append(j)
y=val_list1
cm=tf.math.confusion_matrix(
    y,
    pred_labels_40p,
    
    num_classes=10
)
print("Accuracy 40p : ",accuracy_score(y, pred_labels_40p))
plt.clf()
sns_plot=sns.heatmap(cm,xticklabels=class_names,yticklabels=class_names, annot=True, fmt='g')
# sns_plot.figure.savefig("/home/pbenga2s/RnD/Imagenette_OOD/MobileNet/patch/class1/cf_40p.png")
plt.tight_layout()
plt.xlabel('Prediction')
plt.ylabel('Label')
plt.title('Confusion Matrix')
plt.savefig("/home/pbenga2s/RnD/Imagenette_UE/MobileNet/patch/class9/cf_40p.png",bbox_inches='tight')
plt.savefig("/home/pbenga2s/RnD/Imagenette_UE/MobileNet/patch/class9/cf_40p.pdf",bbox_inches='tight')

pred_labels_35p=np.argmax(model.predict(patched_images_35p),axis=-1)
# y = np.concatenate([y for x,y in validation_ds],axis=0)
val_list=[]
for a,b in ds_test:
  b=np.argmax(b, axis=1)
  val_list.append(b)
val_list1=[]
for i in val_list:
  for j in i:
    val_list1.append(j)
y=val_list1
cm=tf.math.confusion_matrix(
    y,
    pred_labels_35p,
    
    num_classes=10
)
print("Accuracy 35p : ",accuracy_score(y, pred_labels_35p))
plt.clf()
sns_plot=sns.heatmap(cm,xticklabels=class_names,yticklabels=class_names, annot=True, fmt='g')
# sns_plot.figure.savefig("/home/pbenga2s/RnD/Imagenette_OOD/MobileNet/patch/class1/cf_35p.png")
plt.tight_layout()
plt.xlabel('Prediction')
plt.ylabel('Label')
plt.title('Confusion Matrix')
plt.savefig("/home/pbenga2s/RnD/Imagenette_UE/MobileNet/patch/class9/cf_35p.png",bbox_inches='tight')
plt.savefig("/home/pbenga2s/RnD/Imagenette_UE/MobileNet/patch/class9/cf_35p.pdf",bbox_inches='tight')

pred_labels_25p=np.argmax(model.predict(patched_images_25p),axis=-1)
# y = np.concatenate([y for x,y in validation_ds],axis=0)
val_list=[]
for a,b in ds_test:
  b=np.argmax(b, axis=1)
  val_list.append(b)
val_list1=[]
for i in val_list:
  for j in i:
    val_list1.append(j)
y=val_list1
cm=tf.math.confusion_matrix(
    y,
    pred_labels_25p,
    
    num_classes=10
)
print("Accuracy 25p : ",accuracy_score(y, pred_labels_25p))
plt.clf()
sns_plot=sns.heatmap(cm,xticklabels=class_names,yticklabels=class_names, annot=True, fmt='g')
# sns_plot.figure.savefig("/home/pbenga2s/RnD/Imagenette_OOD/MobileNet/patch/class1/cf_25p.png")
plt.tight_layout()
plt.xlabel('Prediction')
plt.ylabel('Label')
plt.title('Confusion Matrix')
plt.savefig("/home/pbenga2s/RnD/Imagenette_UE/MobileNet/patch/class9/cf_25p.png",bbox_inches='tight')
plt.savefig("/home/pbenga2s/RnD/Imagenette_UE/MobileNet/patch/class9/cf_25p.pdf",bbox_inches='tight')

pred_labels_20p=np.argmax(model.predict(patched_images_20p),axis=-1)
# y = np.concatenate([y for x,y in validation_ds],axis=0)
val_list=[]
for a,b in ds_test:
  b=np.argmax(b, axis=1)
  val_list.append(b)
val_list1=[]
for i in val_list:
  for j in i:
    val_list1.append(j)
y=val_list1
cm=tf.math.confusion_matrix(
    y,
    pred_labels_20p,
    
    num_classes=10
)
print("Accuracy 20p : ",accuracy_score(y, pred_labels_20p))
plt.clf()
sns_plot=sns.heatmap(cm,xticklabels=class_names,yticklabels=class_names, annot=True, fmt='g')
# sns_plot.figure.savefig("/home/pbenga2s/RnD/Imagenette_OOD/MobileNet/patch/class1/cf_20p.png")
plt.tight_layout()
plt.xlabel('Prediction')
plt.ylabel('Label')
plt.title('Confusion Matrix')
plt.savefig("/home/pbenga2s/RnD/Imagenette_UE/MobileNet/patch/class9/cf_20p.png",bbox_inches='tight')
plt.savefig("/home/pbenga2s/RnD/Imagenette_UE/MobileNet/patch/class9/cf_20p.pdf",bbox_inches='tight')