from ClassificationLayer import build_classificationLayer
from readData import readH5file
from keras.optimizers import RMSprop

classifierLayer=build_classificationLayer()
classifierLayer.compile(loss='sparse_categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])
(x_train, y_train, _), (_,_,_), (_,_,_) = readH5file()

history = classifierLayer.fit(x_train, y_train,
                    batch_size=1024,
                    epochs=30,
                    verbose=1,
                    validation_split=20)
classifierLayer.save('./models/classifierLayer.h5');

