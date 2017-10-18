import tensorflow as tf 
from readRecord import readRecord
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

batch_size = 5
training_steps = 10
FILEPATH_IMAGE = "car.jpg"
# xmin,ymin,xmax,ymax,Frame,Label,Preview URL
#785,533,905,644,1479498371963069978.jpg,Car,http://crowdai.com/images/Wwj-gorOCisE7uxA/visualize
image, labels, bbox = readRecord(batch_size)

def _filename_queue(path):
  return tf.train.string_input_producer([path])

def createBBoxes(bboxwhole, elem):
  YMAX = 3
  XMIN = 0
  XMAX = 2
  YMIN = 1
  full = bboxwhole
  offset_height = full[elem][YMIN]
  offset_width = full[elem][XMIN]
  target_height = full[elem][YMAX] - offset_height
  target_width = full[elem][XMAX] - offset_width
  
  return offset_height, offset_width, target_height, target_width, full

def cropbb(image, bounding_box):
  return tf.image.crop_to_bounding_box(
    image,
    offset_height = bounding_box[0],
    offset_width = bounding_box[1],
    target_height = bounding_box[2],
    target_width = bounding_box[3]
)
def pad_croppedbb(image):
  return tf.image.pad_to_bounding_box(
    image,
    offset_height = 0,
    offset_width = 0,
    target_height = 300,
    target_width = 480
)

def get_bb_batch(image, bbox_batch, batch_size):
  for elem in range(batch_size):
    #oh, ow, th, tw, full= createBBoxes(bbox_batch, elem)
    cropped_image = pad_croppedbb(cropbb(image, createBBoxes(bbox_batch, elem)))
    #print(cropped_image)
    return cropped_image


with tf.Session() as sess:
  tf.global_variables_initializer().run()
  # Coordinate the loading of image files.
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)

  for step in range(training_steps):
    get_bb_batch(image, bbox, batch_size)



#  for step in range(training_steps):
#      batch_of_BB = sess.run(bbox)
#      for elem in range(batch_size):
#        oh, ow, th, tw, full= createBBoxes(batch_of_BB, elem)
#        cropped_image = pad_croppedbb(cropbb(image, oh,ow, th, tw))
#
        #print(oh,ow, th, tw, full)
#        print("cropped", cropped_image)


 
   
  coord.request_stop()
  coord.join(threads)
  sess.close()