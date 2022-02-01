import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import numpy as np
from datetime import datetime
import sklearn
import tensorboard
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import  confusion_matrix, accuracy_score
#tf.compat.v1.disable_eager_execution()
tf.executing_eagerly()



# img_height = 500
# img_width = 500
# batch_size = 32
# ds_train = tf.keras.preprocessing.image_dataset_from_directory(
#     "/home/pbenga2s/RnD/Beans/Beans/train",
#     labels="inferred",
#     label_mode="categorical",  
#     color_mode="rgb",
#     batch_size=batch_size,
#     image_size=(img_height, img_width), 
#     shuffle=True,
#     seed=123,
# )
# ds_test = tf.keras.preprocessing.image_dataset_from_directory(
#     "/home/pbenga2s/RnD/Beans/Beans/test",
#     labels="inferred",
#     label_mode="categorical",
#     color_mode="rgb",
#     batch_size=batch_size,
#     image_size=(img_height, img_width),
#     shuffle=True,
#     seed=123,
# )

(ds_train,ds_test),ds_info = tfds.load(
    'imagenette/160px', 
    split=['train','validation'], 
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
    )
label_names = ds_info.features['label'].names
print (label_names)


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

class BayesRiskMSE(keras.losses.Loss):
    """
    Args:
      pos_weight: Scalar to affect the positive labels of the loss function.
      weight: Scalar to affect the entirety of the loss function.
      from_logits: Whether to compute loss form logits or the probability.
      reduction: Type of tf.keras.losses.Reduction to apply to loss.
      name: Name of the loss function.
    """
    def __init__(self, annealing_step,
                 name='bayes_risk'):
        super(BayesRiskMSE, self).__init__(name=name) 
        self.global_step = 0
        self.annealing_step = annealing_step

    def KL(self, alpha, beta):
        sum_alpha = tf.reduce_sum(input_tensor=alpha,axis=1,keepdims=True)
        sum_beta = tf.reduce_sum(input_tensor=beta,axis=1,keepdims=True)
        lnB = tf.math.lgamma(sum_alpha) - tf.reduce_sum(input_tensor=tf.math.lgamma(alpha),axis=1,keepdims=True)
        lnB_uni = tf.reduce_sum(input_tensor=tf.math.lgamma(beta),axis=1,keepdims=True) - tf.math.lgamma(sum_beta)
        dg0 = tf.math.digamma(sum_alpha)
        dg1 = tf.math.digamma(alpha)
        kl = tf.reduce_sum(input_tensor=(alpha - beta)*(dg1-dg0),axis=1,keepdims=True) + lnB + lnB_uni
        return kl

    def call(self, y_true, alpha):
        self.global_step += 1
        #alpha = alpha + 1
        S = tf.reduce_sum(input_tensor=alpha, axis=1, keepdims=True) #Dirichlet strength
        evidence = alpha - 1 #evidence 
        p_k = alpha / S # Eq 2, expected probability for the kth singleton
        y_true=tf.cast(y_true, tf.float32)
        A = tf.reduce_sum(input_tensor=(y_true-p_k)**2, axis=1, keepdims=True)
        B = tf.reduce_sum(input_tensor=alpha*(S-alpha)/(S*S*(S+1)), axis=1, keepdims=True)
        #annealing_coef = 0.01
        annealing_coef = tf.minimum(1.0,tf.cast(self.global_step/self.annealing_step,tf.float32))
        # tf.minimum(1.0,tf.cast(global_step/annealing_step,tf.float32))
        #ToDo annealing_step is number of batched, global step is initial 0
        updated_alpha = evidence*(1-y_true) + 1
        #num_of_classes = tf.shape(y_true)[-1].value
        #print ("num_of_classes ", num_of_classes)
        beta=tf.constant(np.ones((1, 10)),dtype=tf.float32)
        C = annealing_coef * self.KL(updated_alpha, beta)
        loss = tf.reduce_mean(A+B+C)
        return loss
#model.compile(optimizer=keras.optimizers.Adam(),
#              loss=WeightedBinaryCrossEntropy(0.5, 2))
#model.fit(x_train, y_train, batch_size=64, epochs=3)
def custom_accuracy(y_true, alpha):
    # Calculate accuracy
    pred = tf.argmax(input=alpha, axis=1)
    truth = tf.argmax(input=y_true, axis=1)
    match = tf.reshape(tf.cast(tf.equal(pred, truth), tf.float32),(-1,1))
    acc = tf.reduce_mean(input_tensor=match)
    return acc

def mean_evidence_success(y_true, alpha):
    pred = tf.argmax(input=alpha, axis=1)
    truth = tf.argmax(input=y_true, axis=1)
    match = tf.reshape(tf.cast(tf.equal(pred, truth), tf.float32),(-1,1))
    evidence = alpha - 1 #evidence 
    mean_ev_succ = tf.reduce_sum(input_tensor=tf.reduce_sum(input_tensor=evidence,axis=1, keepdims=True)*match) / tf.reduce_sum(input_tensor=match+1e-20)
    return mean_ev_succ

def mean_evidence_failure(y_true, alpha):
    pred = tf.argmax(input=alpha, axis=1)
    truth = tf.argmax(input=y_true, axis=1)
    match = tf.reshape(tf.cast(tf.equal(pred, truth), tf.float32),(-1,1))
    evidence = alpha - 1 #evidence 
    mean_ev_fail = tf.reduce_sum(input_tensor=tf.reduce_sum(input_tensor=evidence,axis=1, keepdims=True)*(1-match)) / (tf.reduce_sum(input_tensor=tf.abs(1-match))+1e-20)
    return mean_ev_fail



preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip("horizontal"),
        layers.experimental.preprocessing.RandomRotation(0.1),
    ]
)
model=tf.keras.applications.MobileNetV2(
    input_shape=(160,160,3), alpha=0.35,include_top=False,
    weights='imagenet',  classes=10
)
model.trainable = True
inputs = tf.keras.Input(shape=(160, 160, 3))
# x = data_augmentation(inputs)
x = preprocess_input(inputs)
x = model(x, training=True)
# x=layers.Conv2D(32, 3, activation='relu')(x)
# x=layers.MaxPooling2D()(x)
x=layers.Conv2D(32, 3, activation='relu')(x)
x=layers.MaxPooling2D()(x)
x=layers.Flatten()(x)
x=layers.Dense(1280, activation='relu')(x)
x=layers.Dense(10, activation='relu', kernel_initializer='random_normal',
    bias_initializer='ones')(x)
model = keras.Model(inputs=inputs, outputs=x)



# Specify the training configuration (optimizer, loss, metrics)
model.compile(optimizer=keras.optimizers.Adam(0.0001),  # Optimizer
              # Loss function to minimize
              loss=BayesRiskMSE(annealing_step=4),
              #loss=keras.losses.CategoricalCrossentropy())
              # List of metrics to monitor
              metrics=['accuracy', custom_accuracy, mean_evidence_success, mean_evidence_failure])
# Train the model by slicing the data into "batches"
# of size "batch_size", and repeatedly iterating over
# the entire dataset for a given number of "epochs"
print('# Fit model on training data')
history = model.fit(ds_train, 
                    epochs=25,
                    # We pass some validation for
                    # monitoring validation loss and metrics
                    # at the end of each epoch
                    validation_data=ds_test)



model.evaluate(ds_test)



model.predict(ds_test)

export_path = "/home/pbenga2s/RnD/Imagenette_UE/saved_models/MobileNet/MobileNetV2_model(1)"
model.save(export_path)

export_path

class_names=label_names
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
plt.tight_layout()
plt.xlabel('Prediction')
plt.ylabel('Label')
plt.title('Confusion Matrix')
plt.savefig("/home/pbenga2s/RnD/Imagenette_UE/MobileNet/cf_clean.png",bbox_inches='tight')
plt.savefig("/home/pbenga2s/RnD/Imagenette_UE/MobileNet/cf_clean.pdf",bbox_inches='tight')