from keras.layers import Conv2D, MaxPool2D,\
 GlobalAveragePooling2D,Dense,\
 Input, Reshape, UpSampling2D,\
 BatchNormalization, GaussianNoise
from keras.models import Model
from keras.optimizers import Adam
act_func = 'selu'
aec_dim_num = 2
encoder_layers =[GaussianNoise(1),
 BatchNormalization(),
 Conv2D(32, (7,7),padding = 'same',
 activation=act_func),
MaxPool2D(2,2),
BatchNormalization(),
Conv2D(64, (5,5),padding = 'same',
activation=act_func),
MaxPool2D(2,2),
BatchNormalization(),]