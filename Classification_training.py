import os
import sys
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras import callbacks
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras import backend as K
import tensorflow as tf
import numpy as np
import scipy.io as sio
import time

def main():
    start_time = time.time()
    input_base = '/srv/home/lerandc/outputs/712_STO/'
    input_sub_folder = ['0_0/','05_0/','025_025/','1_0/','1_1/','2_0/','2_2/','3_0/']    
    result_path =  '/srv/home/lerandc/CNN/models/072418_vgg16_softmax/attempt2/'

    x_train_list = []
    y_train_list = []

    sx, sy = 0, 0

    for current_folder in input_sub_folder:
        input_folder = input_base + current_folder
        input_images = [image for image in os.listdir(input_folder) if 'Sr_PACBED' in image]

        for image in input_images:
            cmp = image.split('_')
            if ('noise' in image):
                label = int(cmp[-2][:])
            else:
                label = int(cmp[-1][:-4])  
                 
            if (('noise100' in image)):

                img = np.load(input_folder + image).astype(dtype=np.float64)
                img = scale_range(img,0,1)
                img = img.astype(dtype=np.float32)
                img_size = img.shape[0]
                sx, sy = img.shape[0], img.shape[1]
                new_channel = np.zeros((img_size, img_size))
                img_stack = np.dstack((img, new_channel, new_channel))

                x_train_list.append(img_stack)
                y_train_list.append(label)

    nb_train_samples = len(x_train_list)
    print('Image loaded')
    print('input shape: ')
    print(sx, sy)
    print('training number: ')
    print(nb_train_samples)
    nb_class = len(set(y_train_list))
    x_train = np.concatenate([arr[np.newaxis] for arr in x_train_list])
    y_train = to_categorical(y_train_list, num_classes=nb_class)
    print('Size of image array in bytes')
    print(x_train.nbytes)
    np.save(result_path + 'y_train.npy', y_train)


    logs = [log for log in os.listdir(result_path) if 'log' in log]
    max_index = 0
    for log in logs:
        cur = int(log.split('_')[1])
        if cur > max_index:
            max_index = cur
    max_index = max_index + 1

    batch_size = 32
    # step 1
    save_bottleneck_features(x_train, y_train, batch_size, nb_train_samples,result_path)

    # step 2
    epochs = 12
    batch_size = 32  # batch size 32 works for the fullsize simulation library which has 19968 total files, total number of training file must be integer times of batch_size
    train_top_model(y_train, nb_class, max_index, epochs, batch_size, input_folder, result_path)

    # step 3
    epochs = 50
    batch_size = 32
    fine_tune(x_train, y_train, sx, sy, max_index, epochs, batch_size, input_folder, result_path)

    print('Total computing time is: ')
    print(int((time.time() - start_time) * 100) / 100.0)


def save_bottleneck_features(x_train, y_train, batch_size, nb_train_samples,result_path):
    model = applications.VGG16(include_top=False, weights='imagenet')
    print('before featurewise center')
    
    datagen = ImageDataGenerator(
        featurewise_center=True,
        rotation_range=90,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=1,
        vertical_flip=1,
        shear_range=0.05)
 

    datagen = ImageDataGenerator(
        featurewise_center=True)

    datagen.fit(x_train)
    print('made it past featurewise center')
    generator = datagen.flow(
        x_train,
        y_train,
        batch_size=batch_size,
        shuffle=False)
    print('made it past generator')

    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size)
    print('made it past the bottleneck features')
    np.save(result_path + 'bottleneck_features_train.npy',
            bottleneck_features_train)

def train_top_model(y_train, nb_class, max_index, epochs, batch_size, input_folder, result_path):
    train_data = np.load(result_path + 'bottleneck_features_train.npy')
    train_labels = y_train
    print(train_data.shape, train_labels.shape)
    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(nb_class, activation='softmax'))

    # compile setting:
    lr = 0.001
    decay = 1e-6
    momentum = 0.9
    optimizer = optimizers.SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
    loss = 'categorical_crossentropy'
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    
    bottleneck_log = result_path + 'training_' + str(max_index) + '_bnfeature_log.csv'
    csv_logger_bnfeature = callbacks.CSVLogger(bottleneck_log)
    earlystop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=3, verbose=1, mode='auto')

    model.fit(train_data,train_labels,epochs=epochs,batch_size=batch_size,shuffle=True,
            callbacks=[csv_logger_bnfeature, earlystop],verbose=2,validation_split=0.2)

    with open(bottleneck_log, 'a') as log:
        log.write('\n')
        log.write('input images: ' + input_folder + '\n')
        log.write('batch_size:' + str(batch_size) + '\n')
        log.write('learning rate: ' + str(lr) + '\n')
        log.write('learning rate decay: ' + str(decay) + '\n')
        log.write('momentum: ' + str(momentum) + '\n')
        log.write('loss: ' + loss + '\n')

    model.save_weights(result_path + 'bottleneck_fc_model.h5')

def fine_tune(train_data, train_labels, sx, sy, max_index, epochs, batch_size, input_folder, result_path):
    print(train_data.shape, train_labels.shape)

    model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(sx, sy, 3))
    print('Model loaded')

    top_model = Sequential()
    top_model.add(Flatten(input_shape=model.output_shape[1:]))
    top_model.add(Dropout(0.3))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.3))
    top_model.add(Dense(52, activation='softmax'))

    top_model.load_weights(result_path + 'bottleneck_fc_model.h5')

    new_model = Sequential()
    for l in model.layers:
        new_model.add(l)
    new_model.add(top_model)

    # compile settings
    lr = 0.0001
    decay = 1e-6
    momentum = 0.9
    optimizer = optimizers.SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
    loss = 'categorical_crossentropy'
    new_model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    fineture_log = result_path + 'training_' + str(max_index) + '_finetune_log.csv'
    csv_logger_finetune = callbacks.CSVLogger(fineture_log)
    earlystop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=5, verbose=1, mode='auto')

    datagen = ImageDataGenerator(
        featurewise_center=True,
        rotation_range=90,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=1,
        vertical_flip=1,
        shear_range=0.05)

    datagen.fit(train_data)

    generator = datagen.flow(
        train_data,
        train_labels,
        batch_size=batch_size,
        shuffle=True)

    validation_generator = datagen.flow(
        train_data,
        train_labels,
        batch_size=batch_size,
        shuffle=True)

    new_model.fit_generator(generator,epochs=epochs,steps_per_epoch=len(train_data) / 32,validation_data=validation_generator,validation_steps=(len(train_data)//5)//32,
            callbacks=[csv_logger_finetune, earlystop],verbose=2)

    #new_model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size, shuffle=True, validation_split=0.2,
                  #callbacks=[csv_logger_finetune, earlystop])

    with open(fineture_log, 'a') as log:
        log.write('\n')
        log.write('input images: ' + input_folder + '\n')
        log.write('batch_size:' + str(batch_size) + '\n')
        log.write('learning rate: ' + str(lr) + '\n')
        log.write('learning rate decay: ' + str(decay) + '\n')
        log.write('momentum: ' + str(momentum) + '\n')
        log.write('loss: ' + loss + '\n')

    new_model.save(result_path + 'FinalModel.h5')  # save the final model for future loading and prediction


def scale_range (input, min, max):
    input += -(np.min(input))
    input /= np.max(input) / (max - min)
    input += min
    return input    

# step 4 make predictions using experiment results

if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]=str(sys.argv[1])
    main()
