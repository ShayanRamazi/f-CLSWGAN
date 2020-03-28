from ClassificationLayer import build_classificationLayer
from readData import readH5file2
import keras
from keras.layers import Input, Dense,Dropout
from keras.models import Sequential, Model
(x_train, y_train, _), (test_x, test_y,_) = readH5file2()
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, 50)
test_y = keras.utils.to_categorical(test_y, 50)
# model = Sequential()
# model.add(Dense(1024, activation='relu', input_shape=(2048,)))
# model.add(Dropout(0.8))
# model.add(Dense(50, activation='softmax'))
#
# model.summary()
model=build_classificationLayer()
model.compile(loss='categorical_crossentropy',
				  optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.5, beta_2=0.999, amsgrad=False),
				  metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=256,
                    epochs=15,
                    verbose=1,
                    validation_data=(test_x, test_y))
score = model.evaluate(test_x, test_y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
model.save('./models/classifierLayer.h5');

