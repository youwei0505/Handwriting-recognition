
# coding: utf-8

# In[1]:


from __future__ import print_function
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)


# In[2]:


#parameters
learning_rate = 0.1
num_steps = 500
batch_size = 128
display_step = 100
model_path = "/temp/perceptron_model.ckpt"


# In[3]:


#Network Parameters
n_hidden_1 = 256 # number of neurons
n_hidden_2 = 256
num_input = 784 # 28*28
num_classes = 10 #0~9


# In[4]:


#tf Graphic input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])


# In[5]:


# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2,num_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out' : tf.Variable(tf.random_normal([num_classes]))
}


# In[6]:


# build percetron model
def neural_net(x):
    #Fully connected hidden layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer


# In[7]:


#Construct model
logits = neural_net(X)
prediction = tf.nn.softmax(logits)

#Evaluate model
#Argmax returns the index with the largest value across axes of a tensor
ans = tf.argmax(prediction, 1)


# In[8]:


#Loss function & optimizer
loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_function)


# In[9]:


#Evaluate model
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[10]:


init = tf.global_variables_initializer()


# In[11]:


#Save & Restore variables
saver = tf.train.Saver()


# In[12]:


# Training
with tf.Session() as sess:
    sess.run(init)
    
    for step in range(1, num_steps+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict={X: batch_x, Y:batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_function,accuracy],feed_dict={X: batch_x,
                                                                    Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " +                   "{:.4f}".format(loss) + ", Traning Acuuracy= " +                  "{:.3f}".format(acc))
            
    print("Done!")
    
    #Save
    save_path = saver.save(sess, model_path)
    print("Model restores from file: %s" % save_path)
    #Caculate accuracy
    print("Testing Accuracy:",           sess.run(accuracy, feed_dict={X: mnist.test.images,
                                        Y: mnist.test.labels}))
    
#Running test database by "loading" the model saved earlier
with tf.Session() as sess:
    sess.run(init)
    
    saver.restore(sess, model_path)
    print("Model restores from file: %s" % save_path)
    
    # Calculate accuracy 
    print("Testing Accuracy:",           sess.run(accuracy, feed_dict={X: mnist.test.images,
                                        Y: mnist.test.labels}))

