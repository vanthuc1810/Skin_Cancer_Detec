import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet import ResNet50
from keras.layers import Flatten, Input, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Dense
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import os

BATCH_SIZE = 64
data_dir = 'hmnist_28_28_RGB.csv'
data = pd.read_csv(data_dir)
#tron du lieu
data_shuf = data.sample(frac=1, random_state=42).reset_index(drop=True)


y = data_shuf['label']
x = data_shuf.drop(columns='label')

oversample = RandomOverSampler()
x, y = oversample.fit_resample(x,y)

#reshape anh ve RGB
x = np.array(x).reshape(-1,28,28,3)
y = np.array(y)


X_train, X_val, y_train, y_val = train_test_split(x,y,random_state=42,test_size=0.2)
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)


#Xay dung model

input_tensor = Input(shape=(28,28,3)) # dau vao
base_model = ResNet50(weights='imagenet', include_top=False)
for layer in base_model.layers:
    layer.trainable = False
x = base_model(input_tensor)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
x = Dense(25, activation='softmax',name='predict')(x)
model = Model(inputs=input_tensor,outputs = x)
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

aug = ImageDataGenerator(
    rescale= (1./255),
    rotation_range=40,
    zoom_range = 0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip = True
)

aug_val = ImageDataGenerator(rescale=1./255)

#Tao callback
checkpoint_dir = ''
filepath = os.path.join(checkpoint_dir, "weights-{epoch:02d}-{val_accuracy:.2f}.hdf5")
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]


H = model.fit(aug.flow(X_train,y_train,batch_size=BATCH_SIZE),
              epochs=50,
              validation_data=aug.flow(X_val,y_val,batch_size=BATCH_SIZE),
              verbose=1,
              callbacks=callbacks_list)

