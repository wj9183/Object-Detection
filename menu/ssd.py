import streamlit as st
import cv2
import tensorflow as tf
import os
import pathlib
import keras

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image


from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile

# 내 로컬에 설치된 TFOD 경로
PATH_TO_LABELS = "./models/ssd_mobilenet_v2_320x320_coco17_tpu-8/saved_model/mscoco_label_map.pbtxt"
# with open("./models/ssd_mobilenet_v2_320x320_coco17_tpu-8/mscoco_label_map.pbtxt", 'rb') as file:
#     st.write(file)
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
# st.write(category_index)



def load_image(image_file):
    img = Image.open(image_file)
    return img

def run_inference_for_single_image(model, image):
  # 넘파이 어레이로 바꿔준다.
  image = np.asarray(image)
  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  input_tensor = tf.convert_to_tensor(image)
  # The model expects a batch of images, so add an axis with `tf.newaxis`.
  input_tensor = input_tensor[tf.newaxis,...]

  # Run inference
  model_fn = model.signatures['serving_default']
  output_dict = model_fn(input_tensor)

  # All outputs are batches tensors.
  # Convert to numpy arrays, and take index [0] to remove the batch dimension.
  # We're only interested in the first num_detections.
  num_detections = int(output_dict.pop('num_detections'))
  output_dict = {key:value[0, :num_detections].numpy() 
                 for key,value in output_dict.items()}

  # print(output_dict)
  
  output_dict['num_detections'] = num_detections

  # detection_classes should be ints.
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
   
  # Handle models with masks:
  if 'detection_masks' in output_dict:
    output_dict['detection_masks'] = tf.convert_to_tensor(output_dict['detection_masks'], dtype=tf.float32)
    output_dict['detection_boxes'] = tf.convert_to_tensor(output_dict['detection_boxes'], dtype=tf.float32)
    # Reframe the the bbox mask to the image size.
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
              output_dict['detection_masks'], output_dict['detection_boxes'],
               image.shape[0], image.shape[1])  
    detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                       tf.uint8)
    output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    
  return output_dict





def show_inference(model, image_path):
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.

    image_np = np.array(Image.open(image_path))
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # image_np = cv2.imread(str(image_path))
    # print(image_np)
    # Actual detection.
    output_dict = run_inference_for_single_image(model, image_np)
    # Visualization of the results of a detection.
    print(output_dict)
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.array(output_dict['detection_boxes']),
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks_reframed',None),
        use_normalized_coordinates=True,
        line_thickness=8)
    image_np_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    scaleX = 0.6
    scaleY = 0.6
    image_np_bgr_resize = cv2.resize(image_np_bgr, None, fx = scaleX, fy=scaleY, interpolation = cv2.INTER_LINEAR)
    # st.write(image_np_bgr_resize)
    # st.image(cv2.imshow("hand_{}".format(image_path), image_np_bgr_resize))
    st.image(image_np_bgr_resize)






def ssd():
    # st.title("")

    model = keras.models.load_model("./models/ssd_mobilenet_v2_320x320_coco17_tpu-8/saved_model/")

    image_file = st.file_uploader("이미지 파일 업로드", type = ['png', 'jpeg', 'jpg'], accept_multiple_files=False)
 

    
    if image_file is not None:

      show_inference(model, image_file)

    # st.write(model)
    # PATH_TO_TEST_IMAGE_DIR = pathlib.Path('data/images')
    # TEST_IMAGE_PATH = sorted(  list(PATH_TO_TEST_IMAGE_DIR.glob("*.jpg"))  )
    # for image_path in TEST_IMAGE_PATH :
    #     print(image_path)
    #     show_inference(model, image_path)


    # cv2.waitKey()
    # cv2.destroyAllWindows()