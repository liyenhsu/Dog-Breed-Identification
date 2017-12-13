from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np

def load_dataset(path, shuffle=True):
    data = load_files(path, shuffle=shuffle)
    files = data['filenames']
    labels = np_utils.to_categorical(data['target'], 120) # There are 120 dog breeds
    return files, labels

# load train and test datasets
train_files, train_labels = load_dataset('train_data')
test_files, _ = load_dataset('test_data', shuffle=False)

# save the labels of the train set
np.save('train_labels.npy', train_labels)


from keras.preprocessing import image                  
from tqdm import tqdm

def path_to_tensor(img_path):
    # load RGB image. Use image size (299, 299) for InceptionResNetV2 model 
    img = image.load_img(img_path, target_size=(299, 299))
    # convert image to 3D tensor with shape (299, 299, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 299, 299, 3)
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)


from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True                 

# pre-process the data for Keras
train_tensors = paths_to_tensor(train_files)
test_tensors = paths_to_tensor(test_files)


# load InceptionResNet V2 model
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
model = InceptionResNetV2(include_top=False, weights='imagenet')

# preprocess the tensors and get bottleneck_features
bottleneck_features_train = model.predict(preprocess_input(train_tensors))
np.save(open('bottleneck_features_train.npy', 'w'), bottleneck_features_train)
bottleneck_features_test = model.predict(preprocess_input(test_tensors))
np.save(open('bottleneck_features_test.npy', 'w'), bottleneck_features_test)
