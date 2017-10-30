import tensorflow as tf 
from readRecord import readRecord
from imageProcessing import get_bb_batch
from datetime import datetime
#Vis graph
now = datetime.utcnow().strftime("%m%d %H%M") 
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

training_steps = 5000
learning_rate = 0.01
batch_size = 5
image_height = 600
image_width = 960
image_channels = 1

train_dataset = "./tfrecord/trainingdata.tfrecord"  
val_dataset = "./tfrecord/valdata.tfrecord"   
test_dataset = "./tfrecord/testdata.tfrecord" 
TRAIN_SIZE = int(5653/batch_size)
TEST_EVAL_SIZE = int(1884/batch_size)
def run_CNN(dataset_type):

  image, labels, bbox = readRecord(batch_size, dataset_type)

  processed_image_batch = get_bb_batch(image, bbox, batch_size)
  #begin Model--
  #float_image_batch = tf.image.convert_image_dtype(processed_image_batch, tf.float32)
  #image_batch = tf.placeholder(tf.float32, shape=(None, 480, 480, 1))

  image_reshaped = tf.reshape(processed_image_batch, [batch_size,image_height,image_width,image_channels]) 


  conv2d_layer_one = tf.contrib.layers.convolution2d(
                                            image_reshaped,
                                            num_outputs=4,
                                            kernel_size=(4,4),
                                            activation_fn=tf.nn.relu,
                                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                                            stride=(2, 2),
                                            trainable=True)

  pool_layer_one = tf.nn.max_pool(conv2d_layer_one,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME')

  conv2d_layer_two = tf.contrib.layers.convolution2d(
                                            pool_layer_one,
                                            num_outputs=8,
                                            kernel_size=(5,5),
                                            activation_fn=tf.nn.relu,
                                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                                            stride=(1, 1),
                                            trainable=True)

  pool_layer_two = tf.nn.max_pool(conv2d_layer_two,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME')


  conv2d_layer_three = tf.contrib.layers.convolution2d(
                                            pool_layer_two,
                                            num_outputs=16,
                                            kernel_size=(5,5),
                                            activation_fn=tf.nn.relu,
                                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                                            stride=(1, 1),
                                            trainable=True)

  pool_layer_three = tf.nn.max_pool(conv2d_layer_three,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME')


  flattened_layer_two = tf.reshape(pool_layer_three,[batch_size, -1])

  fully_connected_layer_one = tf.contrib.layers.fully_connected(flattened_layer_two,
                                                    num_outputs=8, 
                                                    activation_fn=tf.nn.relu)

  dropout_layer_1 = tf.nn.dropout(fully_connected_layer_one, 0.1)

  fully_connected_layer_two = tf.contrib.layers.fully_connected([dropout_layer_1], num_outputs=3) # 0 to 3 !!!! labels 
 
  #end Model
  return fully_connected_layer_two, labels

  #begin helper---
  #print("conv_layer_one", conv2d_layer_one.get_shape())
  #print("pool_layer_one", pool_layer_one.get_shape())
  #print("flattened_layer_two", flattened_layer_two.get_shape())
  #print("fc", fully_connected_layer_two.get_shape())
  #print("fc rank", fully_connected_layer_two )
  #end helper

def loss(dataset_type):
  outputs, labels = run_CNN(dataset_type) 
  onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=3)

  loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits= outputs))
  return loss
def eval(dataset_type):
  outputs, labels = run_CNN(dataset_type) 
  onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=3)
  predicted = tf.cast(tf.argmax(outputs, 1), tf.int32)
  accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, labels),tf.float32))
  return accuracy

train_loss = loss(train_dataset)
#val_loss = loss(val_dataset)
optimise = tf.train.AdamOptimizer(learning_rate).minimize(train_loss)
test_accuracy = eval(test_dataset)

# save model 
saver = tf.train.Saver()
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

#log training
train_mse = tf.summary.scalar('MSE', train_loss)


with tf.Session() as sess:
  tf.global_variables_initializer().run()
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)

  for step in range(training_steps):
    print(step)    
    sess.run(optimise)
    train_summary_str = train_mse.eval()
    file_writer.add_summary(train_summary_str, step)
    #print("train_loss :", sess.run(train_loss))
    #print("val_loss :", sess.run(val_loss))
  sum_acc = 0
  
  for step in range(TEST_EVAL_SIZE):
    sum_acc += sess.run(test_accuracy)
  print("total_acc :", sum_acc/TEST_EVAL_SIZE)
  
  #saver.save(sess, "./tf_logs/model_weights_5k.ckpt")

  

  file_writer.close()

  coord.request_stop()
  coord.join(threads)
  sess.close()


