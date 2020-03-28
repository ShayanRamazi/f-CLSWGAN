import h5py
import numpy as np
def readH5file(dataSet='awa1',type='binary'):
    filename = "./datasets/"+dataSet+"/data.h5"
    file = h5py.File(filename, 'r')
    #==========================================#
    train_x = np.array(file['train']['X'])
    train_a = np.array(file['train']['A'][type])
    train_y = np.array(file['train']['Y'])
    # ==========================================#
    val_x = np.array(file['val']['X'])
    val_y = np.array(file['val']['Y'])
    val_a = np.array(file['val']['A'][type])
    # ==========================================#
    test_x = np.array(file['test']['unseen']['X'])
    test_a = np.array(file['test']['unseen']['A'][type])
    test_y = np.array(file['test']['unseen']['Y'])
    return (train_x, train_y, train_a), (test_x, test_y,test_a), (val_x, val_y,val_a)
def readH5file2(dataSet='awa1',type='binary'):
    filename = "./datasets/"+dataSet+"/data.h5"
    file = h5py.File(filename, 'r')
    #==========================================#
    train_x = np.array(file['train']['X'])
    train_a = np.array(file['train']['A'][type])
    train_y = np.array(file['train']['Y'])
    # ==========================================#
    test_x = np.array(file['test']['seen']['X'])
    test_a = np.array(file['test']['seen']['A'][type])
    test_y = np.array(file['test']['seen']['Y'])
    return (train_x, train_y, train_a), (test_x, test_y,test_a)
def numberOfClass(dataSet='awa1'):
    filename = "./datasets/"+dataSet+"/data.h5"
    file = h5py.File(filename, 'r')
    return np.unique(file['train']["Y"]).shape[0]+np.unique(file['test']['unseen']['Y']).shape[0]

