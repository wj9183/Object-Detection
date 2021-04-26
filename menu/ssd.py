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
import time

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





def show_inference(model, image_file):
  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.

  image_np = np.array(Image.open(image_file))
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


# def save_uploaded_file(directory, file):
#   # 1. 디렉토리가 있는지 확인하여, 없으면 만든다.
#   if not os.path.exists(directory): #존재하지 않으면.
#       os.makedirs(directory)
#   #이제 디렉토리가 있으므로 파일을 저장한다.
#   with open(os.path.join(directory, file.name), 'wb') as f:     #디렉토리 + 파일명으로 
#       f.write(file.getbuffer())
#   return st.success("Saved file : {} in {}".format(file.name, directory))       #업로드에 성공했다고 초록색으로 뜨는 이유가 st.success 이기 때문.




def ssd_image():
      
  model = keras.models.load_model("./models/ssd_mobilenet_v2_320x320_coco17_tpu-8/saved_model/")
  image_file = st.file_uploader("이미지 파일 업로드", type = ['png', 'jpeg', 'jpg'], accept_multiple_files=False)

  if image_file is not None:
    show_inference(model, image_file)






def ssd_video():
      pass
  #보류      

  # model = keras.models.load_model("./models/ssd_mobilenet_v2_320x320_coco17_tpu-8/saved_model/")

  # st.write("으아아비디오")
  # directory = "./data/videos"
  # video_file = st.file_uploader("동영상 파일 업로드", type = ['mp4'], accept_multiple_files=False)
  # # save_uploaded_file(directory, video_file)
  # if video_file is not None:
  #   # st.video(video_file)
  #       #mp4 비디오 파일에서 읽어오는 것
  #   cap = cv2.VideoCapture('data/videos/library1.mp4')
    
  #   #앞으로 교수님이 체크코드를 빼도 우리는 체크를 알아서 해라
  #   if cap.isOpened() == False:
  #       print("Error opening video stream or file")

  #   else:
  #     box = st.empty()
  #     #반복문이 필요한 이유? 비디오는 여러 사진으로 구성되어있으니까.
  #     #여러개니까!
  #     ret, frame = cap.read()
  #     while ret:
  #           #사진을 한장씩 가져와서, True 또는 False, 넘파이어레이를 변수에 저장.
  #         #제대로 사진을 가져왔으면 화면에 표시한다.
  #         if ret == True:
  #             box.image(frame, channels="BGR")
  #             # start_time = time.time()
  #             # show_inference(model, frame)
  #             # cv2.imshow("Frame", frame)
  # #             end_time = time.time()
  # #             print(end_time - start_time)  
  # #             # 키보드에서 esc키를 누르면 exit하라는 것.
  # #             #실무에서 사용안한다. 알 필요 없는데 이건 확인용이니까.
  # #             #분석할 가치가 없고 로직도 상관없다.
  #             if cv2.waitKey(25) & 0xFF == 27:
  #                 break

  #         else:
  #             break 

  #       #데이터베이스의 커서,커넥션 close와 같은 것.
  # cap.release()

  # cv2.destroyAllWindows()