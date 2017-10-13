from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf 

#--- This file will read labels from a .csv and images from a path and write them as a TFRecord (binary).
#Labels file format in csv form dataset
# xmin,xmax,ymin,ymax,Frame,Label,Preview URL
#785,533,905,644,1479498371963069978.jpg,Car,http://crowdai.com/images/Wwj-gorOCisE7uxA/visualize

FILEPATH_LABELS = ".\dataset\labels.csv"
FILEPATH_IMAGE = ".\dataset\*.jpg"
RECORD_DEFAULTS = [[1], [1], [1], [1], [""], [""], [""]]
RECORD_LENGTH = 9423 #total samples in dataset

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _filename_queue(path):
	return tf.train.string_input_producer([path])

def _decode_csv(value,record_defaults):
	return tf.decode_csv(value, record_defaults=RECORD_DEFAULTS)

def _filename_queue_mul(path):
	return tf.train.string_input_producer(tf.train.match_filenames_once(path))

def _rgb_to_greyscale(image):
	rgb_image_float = tf.image.convert_image_dtype(image, tf.float32)
	return tf.image.rgb_to_grayscale(rgb_image_float)

#begin Image Pipeline--
filename_queue_image = _filename_queue_mul(FILEPATH_IMAGE)

image_reader = tf.WholeFileReader()

_, image_file = image_reader.read(filename_queue_image)

image_decoded = tf.image.decode_jpeg(image_file, channels=3, ratio=2)

greyscale_image = _rgb_to_greyscale(image_decoded)
#end

#begin Label Pipeline---´
filename_queue = _filename_queue(FILEPATH_LABELS)

reader = tf.TextLineReader(skip_header_lines=1)

_, value = reader.read(filename_queue, name="labels")

xmin, xmax, ymin, ymax, frame, label, url = _decode_csv(value, record_defaults=RECORD_DEFAULTS)

BBFeatures = tf.stack([xmin, xmax, ymin, ymax])
#end

with tf.Session() as sess:
	
	sess.run(tf.local_variables_initializer())
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)
	
	writer = tf.python_io.TFRecordWriter("./tfrecord/trainingdata9423.tfrecord")
	
	for i in range(RECORD_LENGTH):
		print("wrting record : ", i+1)

		BBoxes, framelabel = sess.run([BBFeatures, label])
		
		rawImage = sess.run(greyscale_image)

		image_bytes = rawImage.tobytes()
		
		# Create TFRecord
		record = tf.train.Example(features=tf.train.Features(feature={
		'xmin': _int64_feature(BBoxes[0]),
		'xmax': _int64_feature(BBoxes[1]),
		'ymin': _int64_feature(BBoxes[2]),
		'ymax': _int64_feature(BBoxes[3]),
		'label': _bytes_feature(framelabel),
		'imageframe': _bytes_feature(image_bytes)
		}))

		writer.write(record.SerializeToString())
	writer.close()
	coord.request_stop()
	coord.join(threads)

