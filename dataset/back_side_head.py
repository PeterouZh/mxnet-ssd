import os
os.sys.path.append('..')
import numpy as np
from imdb import Imdb
from tools.get_file_list import GetFileListOrder
import xml.etree.ElementTree as ET
from evaluate.eval_voc import voc_eval
import cv2

class BackSideHead(Imdb):
    """
     Implementation of Imdb for back side of head datasets
    
    Parameters:
    ----------
     image_set : str
         Subdirectory of images
     mat : str
         Subdirectory of images' annotations or label file
     parent_path : str
         Parent path of image_set and mat
    shuffle : boolean
        whether to initial shuffle the image list
    is_train : boolean
        if true, will load annotations
    """
    def __init__(self, image_set, mat, parent_path, shuffle = True, is_train = False):
        super(BackSideHead, self).__init__(image_set)
        self.image_set = image_set
        if os.path.isdir(os.path.join(parent_path, mat)):
            self.mat_path = mat
        elif os.path.isfile(os.path.join(parent_path, mat)):
            self.mat_file = mat
        else:
            assert 0, "Incorrect mat file : {}".format(mat)
        self.parent_path = parent_path
        self.data_path = os.path.join(parent_path, image_set)
        self.img_extension = '.jpg'
        self.label_extension = '.mat'
        self.is_train = is_train

        self.classes = ['back_side_head']
        self.config = {'padding' : 60}

        self.num_classes = len(self.classes)
        self.image_set_path_list = self._load_image_set_path(self.data_path, shuffle)
        self.num_images = len(self.image_set_path_list)
        assert not (hasattr(self, 'mat_path') and hasattr(self, 'mat_file')), "mat_path and mat_file can't be included at the same time."
        if self.is_train:
            if hasattr(self, 'mat_path'):
                self.labels = self._load_image_labels_from_path()
            elif hasattr(self, 'mat_file'):
                self.labels = self._load_image_labels_from_file()
            else:
                assert 0, "Please input correct mat file or path : {}".format(mat)
        
    def image_path_from_index(self, index):
        """
        load image full path given specified index

        Parameters:
        ----------
        index : int
            index of image requested in dataset

        Returns:
        ----------
        full path of specified image
        """
        image_file = self.image_set_path_list[index]
        assert os.path.exists(image_file), 'Path does not exists: {}'.format(image_file)
        return image_file

    def label_from_index(self, index):
        """
        given image index, return preprocessed ground-truth

        Parameters:
        ----------
        index: int
            index of a specific image
        Returns:
        ----------
        ground-truths of this image
        """
        assert self.labels is not None, "Labels not processed"
        return self.labels[index, :, :]

    def _load_image_set_path(self, imgs_path, shuffle):
        assert os.path.exists(imgs_path), 'Path does not exist: {}'.format(imgs_path)
        image_set_path_list = GetFileListOrder(imgs_path, endswith_flag = True,
                                               end_strs = self.img_extension)
        if shuffle:
            np.random.shuffle(image_set_path_list)
        return image_set_path_list

    def _load_image_labels_from_path(self):
        """
        preprocess all ground-truths

        Returns:
        ----------
        labels packed in [num_images x max_num_objects x 5] tensor
        """
        max_objects = 0
        labels = []
        for image_file in self.image_set_path_list:
            img = cv2.imread(image_file)
            width = img.shape[1]
            height = img.shape[0]
            label_file_name = image_file.split('/')[-1].replace(self.img_extension, self.label_extension)
            label_file = os.path.join(self.parent_path, self.mat_path, label_file_name)
            # test_load_back_side_head_mat(image_file, label_file)
            label = BackSideHead.load_back_side_head_mat(label_file, width, height)
            max_objects = max(max_objects, label.shape[0])
            labels.append(label)
        assert max_objects > 0, "No objects were found for any of the images."
        assert max_objects <= self.config['padding'], "Objects exceed padding."
        self.padding = self.config['padding']
        temp = []
        for label in labels:
            label = np.lib.pad(label, ((0, self.padding - label.shape[0]), (0, 0)),
                               'constant', constant_values = (-1, -1))
            temp.append(label)
        return np.array(temp)

    def _load_image_labels_from_file(self):
        """
        preprocess all ground-truths
        Load all labels from one mat file
        
        Returns:
        ----------
        labels packed in [num_images x max_num_objects x 5] tensor
        """
        max_objects = 0
        labels = []
        label_file = os.path.join(self.parent_path, self.mat_file)
        name_label_dict = BackSideHead.load_back_side_head_total_labels_from_mat(label_file)
        for image_file in self.image_set_path_list:
            img = cv2.imread(image_file)
            img_name = image_file.split('/')[-1]
            width = img.shape[1]
            height = img.shape[0]
            assert name_label_dict.has_key(img_name), "Can't find {}'s label".format(img_name)
            label = name_label_dict[img_name]
            label[:, 1] = label[:, 1] / width # Normalize x and y to the range of [0, 1]
            label[:, 2] = label[:, 2] / height
            label[:, 3] = label[:, 3] / width
            label[:, 4] = label[:, 4] / height
#            test_image_label(image_file, label[:, 1:])
            max_objects = max(max_objects, label.shape[0])
            labels.append(label)
        assert max_objects > 0, "No objects were found for any of the images."
        assert max_objects <= self.config['padding'], "Objects exceed padding."
        self.padding = self.config['padding']
        print "Max number of heads in one image : {},\nPadding : {}".format(max_objects, self.padding)
        temp = []
        for label in labels:
            label = np.lib.pad(label, ((0, self.padding - label.shape[0]), (0, 0)),
                               'constant', constant_values = (-1, -1))
            temp.append(label)
        return np.array(temp)
        

    @staticmethod
    def load_back_side_head_mat(mat_file, img_width, img_height):
        """
         Load annotations mat file of back side of head
        
        Parameters:
        ----------
         mat_file : str
             path of mat file	  
         img_width : int
             normalize the x axis to range of [0, 1]
         img_height : int
             normalize the y axis to range of [0, 1]
        Returns:
        ----------
         labels : numpy.array
             labels of back side of head for one image
             shape : num_object x 5[cls_id, xmin, ymin, xmax, ymax]
        """
        import scipy.io as scio         # Read .mat file
        label_mat = scio.loadmat(mat_file, struct_as_record = False)
        label_data_struct = label_mat['label_data_struct'][0, 0]
        face_label = label_data_struct.face_label.flatten() # reduce face_label from 2 dim to 1 dim
        cls_id = 0                                          # Zero represents back side of head
        labels = []
        for one_face in face_label:
            rect = one_face[0, 0].rect[0, 0]
            xmin = float(rect.x[0, 0])
            ymin = float(rect.y[0, 0])
            xmax = float((xmin + rect.width)[0, 0])
            ymax = float((ymin + rect.height)[0, 0])
            n_xmin = float(xmin / img_width)
            n_ymin = float(ymin / img_height)
            n_xmax = float(xmax / img_width)
            n_ymax = float(ymax / img_height)
            labels.append([cls_id, n_xmin, n_ymin, n_xmax, n_ymax])
        return np.array(labels)
    
    @staticmethod
    def load_back_side_head_total_labels_from_mat(label_file):
        """
         Get total labels from one mat file
        
        Parameters:
        ----------
         label_file : str
             Labels file for all images
        Returns:
        ----------
         name_label_dict : dict
             {name : label, ...}
             label : np.array
             label shape : num_object x 5[cls_id, xmin, ymin, xmax, ymax]
        """
        import scipy.io as scio         # Read .mat file
        labels_mat = scio.loadmat(label_file, struct_as_record = False)
        labels_data = labels_mat['data']
        total_labels = labels_data.flatten() # reduce labels_data from 2 dim to 1 dim
        cls_id = 0                                          # Zero represents back side of head
        name_label_dict = {}
        for one_img in total_labels:
            img_file_name = one_img.imageFilename[0].encode('utf-8')
            img_name = img_file_name.split('/')[-1]
            img_labels = one_img.objectBoundingBoxes
            img_labels = img_labels.astype(np.float32)
            img_labels[:, 2] += img_labels[:, 0] # compute xmax : width + xmin
            img_labels[:, 3] += img_labels[:, 1] # compute ymax : height + ymin
            img_labels = np.lib.pad(img_labels, ((0, 0), (1, 0)), 'constant',
                                    constant_values = (cls_id, cls_id)) # Add cls_id to labels
            name_label_dict.update({img_name : img_labels})
#        assert 0, 'Debug'
        return name_label_dict

def test_load_back_side_head_mat(img_file, mat_file):
    """
     Test the function of BackSidehead::load_back_side_head_mat()
    
    Parameters:
    ----------
     img_file : str
         image path 	  
     mat_file : str
         label file for the image that contains positions of some rectangles
    Returns:
    ----------
     none
    """
    img = cv2.imread(img_file)
    width = img.shape[1]
    height = img.shape[0]
    labels = BackSideHead.load_back_side_head_mat(mat_file, width, height)
    for idx in range(labels.shape[0]):
        rect = labels[idx,]
        xmin = int(rect[1] * width)
        ymin = int(rect[2] * height)
        xmax = int(rect[3] * width)
        ymax = int(rect[4] * height)
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 5)
    img = cv2.resize(img, (640, 480))
    cv2.imshow('test', img)
    cv2.waitKey(0)

def test_image_label(image, labels):
    """
     Plot rects in one image for debuging
    
    Parameters:
    ----------
     image : str
         image path	  
     label : np.array
         shape : num_rect x 4[xmin, ymin, xmax, ymax]
    Returns:
    ----------
     None
    """
    img = cv2.imread(image)
    width = img.shape[1]
    height = img.shape[0]
    for idx in range(labels.shape[0]):
        rect = labels[idx,]
        xmin = int(rect[0] * width)
        ymin = int(rect[1] * height)
        xmax = int(rect[2] * width)
        ymax = int(rect[3] * height)
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 5)
    img = cv2.resize(img, (640, 480))
    cv2.imshow('test', img)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        assert 0, "Exit."
    
if __name__ == '__main__':
    img_file = '/home/shhs/usr/data/back_side_head/test.jpg'
    mat_file = '/home/shhs/usr/data/back_side_head/test.mat'
    test_load_back_side_head_mat(img_file, mat_file)

