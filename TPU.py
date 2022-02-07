import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Input
from tensorflow.keras import utils
from tensorflow.keras.datasets import mnist, cifar10
from tensorflow.keras.models import Model
import keras
from keras.applications.resnet import ResNet50
from keras.applications.vgg16 import VGG16
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os


#TPU CONFIGURATION

cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
tf.config.experimental_connect_to_cluster(cluster_resolver)
tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
tpu_strategy = tf.distribute.TPUStrategy(cluster_resolver)

# DATA CONFIGURATION

_URL = 'https://storage.googleapis.com/kaggle-data-sets/453611/2387160/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20220206%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20220206T190205Z&X-Goog-Expires=259199&X-Goog-SignedHeaders=host&X-Goog-Signature=0cd76c58f4eca37d1b9dec1bf559b465e0345f64394d21ebd015642da49f8eb6ee4bd6c33ab573439f23300aa2721a7ab85b979ef1cec1a0a87924a3d851e24304a85e0b557a2cbd1fbd60cad82a7125ec1a28630944663864c48a409719c2ae09fb4bfa8fadccea7dc771fbc3b02fb8d5409b9b0dd8c8357613197dd0c749c345108251be00f89c10ed516590e7499eaeed3e523b7aa54b5657518d10f5ae9301dd99878847fa0494029427ced7bcae8952fb76bd21605fa6b739809aa5d0206bf9b231a01009a2c360323d0c7bb54c2c252203c7a3a888a253e8f00b49a2446e498eba30339f59064b7a09cde3619566224e898fdf390037fdf98e967dd94f'
path_to_zip = tf.keras.utils.get_file('archive.zip', origin=_URL, extract=True)
PATH = os.path.join(os.path.dirname(path_to_zip))

train_dir = os.path.join(PATH, 'train')
test_dir = os.path.join(PATH, 'test')
validation_dir = os.path.join(PATH, 'validation')


train_dataset_generator = ImageDataGenerator()
train_dataset = train_dataset_generator.flow_from_directory(train_dir, target_size=(224,224))

test_dataset_generator = ImageDataGenerator()
test_dataset = train_dataset_generator.flow_from_directory(test_dir, target_size=(224,224))

validation_dataset_generator = ImageDataGenerator()
validation_dataset = train_dataset_generator.flow_from_directory(test_dir, target_size=(224,224))

# new_input = Input(shape=(256,256,3))

# pre_trained_model = VGG16(include_top=False, input_shape=(224,224,3))
pre_trained_model = ResNet50(include_top=False, input_shape=(224,224,3))


model = Sequential()
model.add(pre_trained_model)
# model.add(Flatten())
# model.add(Conv2D(input_shape=(224,224,3), filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(70, activation='softmax'))
# model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
# model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
# model.add(Conv2D(filters=128,kernel_size=(3,3),padding="same", activation="relu"))
# model.add(Conv2D(filters=128,kernel_size=(3,3),padding="same", activation="relu"))
# model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
# model.add(Conv2D(filters=256,kernel_size=(3,3),padding="same", activation="relu"))
# model.add(Conv2D(filters=256,kernel_size=(3,3),padding="same", activation="relu"))
# model.add(Conv2D(filters=256,kernel_size=(3,3),padding="same", activation="relu"))
# model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
# model.add(Conv2D(filters=512,kernel_size=(3,3),padding="same", activation="relu"))
# model.add(Conv2D(filters=512,kernel_size=(3,3),padding="same", activation="relu"))
# model.add(Conv2D(filters=512,kernel_size=(3,3),padding="same", activation="relu"))
# model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
# model.add(Flatten())
# model.add(Dense(4096, activation='relu'))
# model.add(Dense(2048, activation='relu'))
# model.add(Dense(1024, activation='relu'))
# model.add(Dense(512, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(70, activation='softmax'))

opt = Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

model.summary()

epochs = 30

fitted_model = model.fit(x=train_dataset, 
                        epochs=epochs, 
                        steps_per_epoch=100, 
                        validation_steps=10, 
                        validation_data=validation_dataset, 
                        validation_batch_size=int(len(validation_dataset.filenames)*0.15),
                        verbose=1
                    )

# acc = fitted_model.history['accuracy']
# val_acc = fitted_model.history['val_accuracy']

# loss = fitted_model.history['loss']
# val_loss = fitted_model.history['val_loss']

# epochs_range = range(epochs)

# plt.figure(figsize=(8, 8))
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')

# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.show()


model.save('my_model')
# loaded_model = keras.models.load_model('my_model')

# //results = loaded_model.evaluate(test_dataset, batch_size=128)

