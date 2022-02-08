import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten
import keras
from keras.applications.resnet import ResNet50
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
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
validation_dir = os.path.join(PATH, 'valid')

train_dataset_generator = ImageDataGenerator()
train_dataset = train_dataset_generator.flow_from_directory(train_dir, target_size=(224,224))

test_dataset_generator = ImageDataGenerator()
test_dataset = train_dataset_generator.flow_from_directory(test_dir, target_size=(224,224))

validation_dataset_generator = ImageDataGenerator()
validation_dataset = train_dataset_generator.flow_from_directory(test_dir, target_size=(224,224))

# Model
pre_trained_model = ResNet50(include_top=False, input_shape=(224,224,3))

model = Sequential()
model.add(pre_trained_model)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(70, activation='softmax'))

opt = Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

model.summary()

epochs = 25

# Training
fitted_model = model.fit(x=train_dataset, 
                        epochs=epochs, 
                        steps_per_epoch=100, 
                        validation_steps=10, 
                        validation_data=validation_dataset, 
                        validation_batch_size=int(len(validation_dataset.filenames)*0.15),
                        verbose=1
                    )

model.save('my_model')

loaded_model = keras.models.load_model('my_model')

#Evaluation
loaded_model.evaluate(test_dataset)