#implement Critic

#######################################################################

from __future__ import print_function, division
from keras.layers import Input, Dense
from keras.models import Sequential, Model

#######################################################################

def build_classificationLayer(class_dim=50):

    ##########################################
    features_dim  = 2048
    in_shape      = features_dim
    ##########################################

    model = Sequential()
    model.add(Dense(1024, input_dim=in_shape,name="LC1"))
    model.add(Dense(class_dim, activation='softmax',name="LC2"))
    model.summary()

    ##########################################

    feature = Input(shape=(features_dim,),name="INC")
    classes = model(feature)

    return Model(feature, classes)