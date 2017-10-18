import tensorflow as tf 

def readRecord(batch_Size):
  image_height = 600
  image_width = 960
  image_channels = 1

  _min_after_dequeue = 5
  _batch_size = batch_Size
  _capacity = _min_after_dequeue + 5 * _batch_size

  tf_record_filename_queue = tf.train.string_input_producer(["./TFRecord/trainingdata200.tfrecord"])

  tf_record_reader = tf.TFRecordReader()

  _, tf_record_serialized = tf_record_reader.read(tf_record_filename_queue)

  tf_record_features = tf.parse_single_example(tf_record_serialized,
                            features={
                              'xmin': tf.FixedLenFeature([], tf.int64),
                              'xmax': tf.FixedLenFeature([], tf.int64),
                              'ymin': tf.FixedLenFeature([], tf.int64),
                              'ymax': tf.FixedLenFeature([], tf.int64),
                              'label': tf.FixedLenFeature([], tf.string),
                              'imageframe': tf.FixedLenFeature([], tf.string)
                              })
 
  tf_record_image = tf.decode_raw(tf_record_features['imageframe'], tf.float32) #uint8 for rgb without float conversion 
  
  # Reshape the image
  tf_record_image = tf.reshape(tf_record_image,[600, 960, 1])

  # convert class names to a 0 based class index.    
  tf_record_label = tf.to_int32(tf.argmax(tf.to_int32(tf.stack([        
    tf.equal(tf_record_features['label'], ["Car"]),        
    tf.equal(tf_record_features['label'], ["Truck"]),        
    tf.equal(tf_record_features['label'], ["Pedestrian"])    
    ])), 
  0))

  tf_record_bbox = [tf_record_features['xmin'], tf_record_features['xmax'], tf_record_features['ymin'],tf_record_features['ymax']]

  image, label, bbox = tf.train.shuffle_batch([tf_record_image, tf_record_label, tf_record_bbox],
                               batch_size=_batch_size, 
                               capacity=_capacity, 
                               min_after_dequeue=_min_after_dequeue)
  
  return image, label, bbox

#with tf.Session() as sess:
  sess.run(tf.local_variables_initializer())
  # Coordinate the loading of image files.
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)

  training_steps =100
  for step in range(training_steps):
    
    imageoutput, labeloutput = sess.run(readRecord())
    
    print(labeloutput)

  coord.request_stop()
  coord.join(threads)
  sess.close()