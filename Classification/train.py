import tensorflow as tf 
from readtfrecord import readRecord
training_steps = 100
learning_rate = 0.5
batch_size = 5
image_height = 600
image_width = 960
image_channels = 1

image, labels, bbox = readRecord(batch_size)

#begin Model--
float_image_batch = tf.image.convert_image_dtype(image, tf.float32)

image_reshaped = tf.reshape(image, [batch_size,image_height,image_width,image_channels]) 

conv2d_layer_one = tf.contrib.layers.convolution2d(
                                          image,
                                          num_outputs=16,
                                          ksizeernel_size=(4,4),
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
                                          num_outputs=32,
                                          kernel_size=(5,5),
                                          activation_fn=tf.nn.relu,
                                          weights_initializer=tf.contrib.layers.xavier_initializer(),
                                          stride=(1, 1),
                                          trainable=True)

pool_layer_two = tf.nn.max_pool(conv2d_layer_two,
                                  ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1],
                                  padding='SAME')

flattened_layer_two = tf.reshape(pool_layer_two,[batch_size, -1])

fully_connected_layer_one = tf.contrib.layers.fully_connected(flattened_layer_two,
                                                  num_outputs=16, 
                                                  activation_fn=tf.nn.relu)

dropout_layer_1 = tf.nn.dropout(fully_connected_layer_one, 0.1)

fully_connected_layer_two = tf.contrib.layers.fully_connected([dropout_layer_1], num_outputs=3) # 0 to 3 !!!! labels 
#end Model

#begin helper---
print("conv_layer_one", conv2d_layer_one.get_shape())
print("pool_layer_one", pool_layer_one.get_shape())
print("flattened_layer_two", flattened_layer_two.get_shape())
print("fc", fc.get_shape())
print("fc rank", fc )
#end helper


loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits= fully_connected_layer_two))

optimizer= tf.train.AdamOptimizer(learning_rate).minimize(loss)


with tf.Session() as sess:
  tf.global_variables_initializer().run()
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)

  for step in range(training_steps):
    print(step)
    sess.run(optimizer)
    if step % 10 == 0:
      print("loss: ", sess.run([loss]))
    #if step % 100 == 0:
     #saver.save(sess, 'my-model', global_step=step)
    
  #saver.save(sess, 'my-model', global_step=training_steps) 

  #writer = tf.train.SummaryWriter('./my_graph', sess.graph)
  coord.request_stop()
  coord.join(threads)
  sess.close()