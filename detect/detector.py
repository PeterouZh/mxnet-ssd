import mxnet as mx
import os
import sys
import numpy as np
from timeit import default_timer as timer
from dataset.testdb import TestDB
from dataset.iterator import DetIter
from tools.image_processing import resize, transform
from tools.get_file_list import GetFileListOrder
import cv2

class Detector(object):
    """
    SSD detector which hold a detection network and wraps detection API

    Parameters:
    ----------
    symbol : mx.Symbol
        detection network Symbol
    model_prefix : str
        name prefix of trained model
    epoch : int
        load epoch of trained model
    data_shape : int
        input data resize shape
    mean_pixels : tuple of float
        (mean_r, mean_g, mean_b)
    batch_size : int
        run detection with batch size
    ctx : mx.ctx
        device to use, if None, use mx.cpu() as default context
    """
    def __init__(self, symbol, model_prefix, epoch, data_shape, mean_pixels, \
                 batch_size=1, ctx=None):
        self.ctx = ctx
        if self.ctx is None:
            self.ctx = mx.cpu()
        _, args, auxs = mx.model.load_checkpoint(model_prefix, epoch)
        self.mod = mx.mod.Module(symbol, context=ctx)
        self.data_shape = data_shape
        self.mod.bind(data_shapes=[('data', (batch_size, 3, data_shape, data_shape))])
        self.mod.set_params(args, auxs)
        self.data_shape = data_shape
        self.mean_pixels = mean_pixels

    def detect(self, det_iter, show_timer=False):
        """
        detect all images in iterator

        Parameters:
        ----------
        det_iter : DetIter
            iterator for all testing images
        show_timer : Boolean
            whether to print out detection exec time

        Returns:
        ----------
        list of detection results
        """
        num_images = det_iter._size
        if not isinstance(det_iter, mx.io.PrefetchingIter):
            det_iter = mx.io.PrefetchingIter(det_iter)
        start = timer()
        detections = self.mod.predict(det_iter).asnumpy()
        time_elapsed = timer() - start
        if show_timer:
            print "Detection time for {} images: {:.4f} sec".format(
                num_images, time_elapsed)
        result = []
        for i in range(detections.shape[0]):
            det = detections[i, :, :]
            res = det[np.where(det[:, 0] >= 0)[0]]
            result.append(res)
        return result

    def im_detect(self, im_list, root_dir=None, extension=None, show_timer=False):
        """
        wrapper for detecting multiple images

        Parameters:
        ----------
        im_list : list of str
            image path or list of image paths
        root_dir : str
            directory of input images, optional if image path already
            has full directory information
        extension : str
            image extension, eg. ".jpg", optional

        Returns:
        ----------
        list of detection results in format [det0, det1...], det is in
        format np.array([id, score, xmin, ymin, xmax, ymax]...)
        """
        test_db = TestDB(im_list, root_dir=root_dir, extension=extension)
        test_iter = DetIter(test_db, 1, self.data_shape, self.mean_pixels,
                            is_train=False)
        return self.detect(test_iter, show_timer)

    def visualize_detection(self, img, dets, classes=[], thresh=0.6):
        """
        visualize detections in one image

        Parameters:
        ----------
        img : numpy.array
            image, in bgr format
        dets : numpy.array
            ssd detections, numpy.array([[id, score, x1, y1, x2, y2]...])
            each row is one object
        classes : tuple or list of str
            class names
        thresh : float
            score threshold
        """
        import matplotlib.pyplot as plt
        import random
        plt.imshow(img)
        height = img.shape[0]
        width = img.shape[1]
        colors = dict()
        for i in range(dets.shape[0]):
            cls_id = int(dets[i, 0])
            if cls_id >= 0:
                score = dets[i, 1]
                if score > thresh:
                    if cls_id not in colors:
                        colors[cls_id] = (random.random(), random.random(), random.random())
                    xmin = int(dets[i, 2] * width)
                    ymin = int(dets[i, 3] * height)
                    xmax = int(dets[i, 4] * width)
                    ymax = int(dets[i, 5] * height)
                    rect = plt.Rectangle((xmin, ymin), xmax - xmin,
                                         ymax - ymin, fill=False,
                                         edgecolor=colors[cls_id],
                                         linewidth=3.5)
                    plt.gca().add_patch(rect)
                    class_name = str(cls_id)
                    if classes and len(classes) > cls_id:
                        class_name = classes[cls_id]
                    plt.gca().text(xmin, ymin - 2,
                                    '{:s} {:.3f}'.format(class_name, score),
                                    bbox=dict(facecolor=colors[cls_id], alpha=0.5),
                                    fontsize=12, color='white')
        plt.show()

    def detect_and_visualize(self, im_list, root_dir=None, extension=None,
                             classes=[], thresh=0.6, show_timer=False):
        """
        wrapper for im_detect and visualize_detection

        Parameters:
        ----------
        im_list : list of str or str
            image path or list of image paths
        root_dir : str or None
            directory of input images, optional if image path already
            has full directory information
        extension : str or None
            image extension, eg. ".jpg", optional

        Returns:
        ----------

        """
        import cv2
        dets = self.im_detect(im_list, root_dir, extension, show_timer=show_timer)
        if not isinstance(im_list, list):
            im_list = [im_list]
        assert len(dets) == len(im_list)
        for k, det in enumerate(dets):
            img = cv2.imread(im_list[k])
            img[:, :, (0, 1, 2)] = img[:, :, (2, 1, 0)]
            self.visualize_detection(img, det, classes, thresh)

    def detect_using_camera(self, show_timer = False):
        import cv2
        cap = cv2.VideoCapture(1)
        while True:
            ret, frame = cap.read()
            assert ret, "Open camera failed"
            thresh = 0.3
            self.detect_one_image(frame, thresh, show_timer)
            if cv2.waitKey(50) & 0xFF == ord('q'):
                break

    def detect_images_in_path(self, img_dir):
        """
         Detect all images in a path
        
        Parameters:
        ----------
         img_dir : str
             dir for images	  
         
        Returns:
        ----------
         None 
        """
        import cv2
        img_list = GetFileListOrder(img_dir, 'True', '.jpg')
        print ('image dir : {}'.format(img_dir))
        img_list_file = os.path.join(img_dir, 'test.txt')
        print "Write image list in {}".format(img_list_file)
        with open(img_list_file, 'w') as f:
            for img_file in img_list:
                f.write(img_file.split('/')[-1] + '\n')

        for img_file in img_list:
            img = cv2.imread(img_file)
            thresh = 0.3
            self.detect_one_image(img, thresh, show_timer = True)
            if cv2.waitKey() & 0xFF == ord('q'):
                break


    def detect_one_image(self, cv_img, thresh = 0.3, show_timer = False):
        """
         Detect and plot for one image
        
        Parameters:
        ----------
         cv_img : np.array
             Image for opencv format
         thresh : float
             class_pred thresh
        Returns:
        ----------
         None 
        """
        import cv2
        rects = self.get_detected_result(cv_img, thresh, show_timer)
        for i in range(rects.shape[0]):
            color = [255, 0, 0]
            xmin = rects[i, 0]
            ymin = rects[i, 1]
            xmax = rects[i, 2]
            ymax = rects[i, 3]
            cv2.rectangle(cv_img, (xmin, ymin), (xmax, ymax), color, 3)
        cv_img  = cv2.resize(cv_img, (640, 480))
        cv2.imshow('frame', cv_img)

    def save_results_to_file(self, root_dir, img_list_file, image_dir, save_dir):
        """
         Save detected results in file for images in img_list_file
        
        Parameters:
        ----------
         root_dir : str
             root dir for all files
         img_list_file : str
             images list file to be detected	  
         image_dir : str
             total images dir
         save_dir : str
             dir for results to be saved in 
        Returns:
        ----------
         None 
        """
        temp = os.path.join(root_dir, save_dir)
        if not os.path.exists(temp):
            os.makedirs(temp)
        img_list_path = os.path.join(root_dir, img_list_file)
        with open(img_list_path, 'r') as f:
            print "Open {}".format(img_list_path)
            total_images_list = f.readlines()
            num_images = len(total_images_list)
            for count, img_file in enumerate(total_images_list):
                img_name = img_file.split()[0]
                out_rect_file = img_name.replace('.jpg', '.txt')
                img_path = os.path.join(root_dir, image_dir, img_name)
                out_rect_file_path = os.path.join(root_dir, save_dir, out_rect_file)
                img = cv2.imread(img_path)
                rects = self.get_detected_result(img, thresh = 0.3, show_timer = False)
                np.savetxt(out_rect_file_path, rects)
                sys.stdout.write("Write results[{}/{}] in {}\r".format(count, num_images, out_rect_file))
#                assert count <= 20, "Debug"
                
    def get_detected_result(self, cv_img, thresh = 0.3, show_timer = False):
        """
         Get detected results for one image
        
        Parameters:
        ----------
         cv_img : np.array
             Image for opencv format
         thresh : float
             class_pred thresh
        Returns:
        ----------
         rects : np.array
             shape : num_object x 4[xmin, ymin, xmax, ymax]
        """
        import cv2
        frame = cv_img
        width = frame.shape[1]
        height = frame.shape[0]
        data = resize(frame, (self.data_shape, self.data_shape), cv2.INTER_LINEAR)
        data = transform(data, self.mean_pixels)
        mx_img_batch = mx.nd.array([data, ])
        data_batch = mx.io.DataBatch(data = [mx_img_batch], label = [None])
        start = timer()
        self.mod.forward(data_batch, is_train = False)
        time_elapsed = timer() - start
        detections = self.mod.get_outputs()[0].asnumpy()
        det = detections[0, :, :]
        results = det[np.where(det[:, 0] >= 0)[0]]
        rect = []
        for i in range(results.shape[0]):
            score = results[i, 1]
            if score > thresh:
                xmin = int(results[i, 2] * width)
                ymin = int(results[i, 3] * height)
                xmax = int(results[i, 4] * width)
                ymax = int(results[i, 5] * height)
                rect.append([xmin, ymin, xmax, ymax])
        if show_timer:
            print "Detection time : {:.4f} sec".format(time_elapsed)
        return np.array(rect)
