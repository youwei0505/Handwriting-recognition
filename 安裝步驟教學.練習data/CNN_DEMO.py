
# coding: utf-8

# In[1]:


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.examples.tutorials.mnist.input_data as input_data
from time import time
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# In[2]:


#parameters
learningrate=0.0001
trainEpochs = 30
batch_size = 100
totalBatchs = int(mnist.train.num_examples/batch_size)
epoch_list = []
accuracy_list = []
loss_list = []
model_path = "c:/pythonwork/CNN_model/Mnist_CNN_model.ckpt"


# In[3]:


#define weight & bias
def weight(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1),name='W')
def bias(shape):
    return tf.Variable(tf.constant(0.1, shape=shape), name='b')

#define layer
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


# In[4]:


#Input layer
with tf.name_scope('Input_Layer'):
    x = tf.placeholder("float", shape=[None, 784], name='x')
    x_imge = tf.reshape(x, [-1,28,28,1])
    
#Conv_1
with tf.name_scope('C1_Conv'):
    W1 = weight([5,5,1,16])
    b1 = bias([16])
    Conv1 = conv2d(x_imge, W1) + b1
    C1_Conv = tf.nn.relu(Conv1)
    
#Max_pool1
with tf.name_scope('C1_Pool'):
    C1_Pool = max_pool_2x2(C1_Conv)
    
#Conv_2
with tf.name_scope('C2_Conv'):
    W2 = weight([5,5,16,36])
    b2 = bias([36])
    Conv2 = conv2d(C1_Pool, W2) + b2
    C2_Conv = tf.nn.relu(Conv2)
    
#Max_pool2
with tf.name_scope('C2_Pool'):
    C2_Pool = max_pool_2x2(C2_Conv)
    
#Flat layer
with tf.name_scope('D_flat'):
    D_flat = tf.reshape(C2_Pool, [-1, 1764])
    
#Hidden layer
with tf.name_scope('D_Hidden_Layer'):
    W3 = weight([1764, 128])
    b3 = bias([128])
    D_Hidden = tf.nn.relu(tf.matmul(D_flat, W3) + b3)
    D_Hidden_Dropout = tf.nn.dropout(D_Hidden, keep_prob=0.8)
    
#Output layer
with tf.name_scope('Output_layer'):
    W4 = weight([128,10])
    b4 = bias([10])
    y_predict = tf.nn.softmax(tf.matmul(D_Hidden_Dropout, W4) + b4)


# In[5]:


#define Training
with tf.name_scope("optimizer"):
    y_label = tf.placeholder("float", shape=[None, 10], name="y_label")
    loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits
                                   (logits=y_predict, labels = y_label))
    optimizer = tf.train.AdamOptimizer(learning_rate=learningrate).minimize(loss_function)

#Accuracy
with tf.name_scope("evaluate_model"):
    correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y_label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


# In[6]:


epoch_list.clear()
loss_list.clear()
accuracy_list.clear()

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

saver = tf.train.Saver()

for epoch in range(trainEpochs):
    for i in range(totalBatchs):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict={x: batch_x,
                                       y_label: batch_y})
    loss,acc=sess.run([loss_function,accuracy],
                      feed_dict={x: mnist.validation.images,
                                 y_label: mnist.validation.labels})
    epoch_list.append(epoch)
    loss_list.append(loss)
    accuracy_list.append(acc)
    print("Train Epoch:", '%02d' %(epoch+1), "Loss=", "{:.9f}".format(loss),
          "Accuracy=",acc)

save_path = saver.save(sess, model_path)

print("Accuracy:",sess.run(accuracy,feed_dict={x: mnist.validation.images,
                                               y_label: mnist.validation.labels}))
print("Model restores from file: %s" % save_path)


# In[7]:


prediction_test = sess.run(y_predict,feed_dict={x:mnist.test.images,                                                      y_label: mnist.test.labels})
print(prediction_test)
print(len(prediction_test))


# In[8]:


prediction_result =sess.run(tf.argmax(y_predict,1),                            feed_dict={x:mnist.test.images,                                       y_label: mnist.test.labels})
prediction_result[:10]


# In[9]:


#Plot loss_function
get_ipython().run_line_magic('matplotlib', 'inline')
fig = plt.gcf()
fig.set_size_inches(8,4)
plt.plot(epoch_list, loss_list, label='loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss'], loc='upper left')


# In[10]:


plt.plot(epoch_list, accuracy_list, label="accuracy")
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.show


# In[11]:


def plot_images_labels_prediction(images,labels,prediction,idx,num=25):
    fig = plt.gcf()
    fig.set_size_inches(12,14)
    if num>25:
        num=25
    for i in range(0,num):
        ax = plt.subplot(5,5, 1+i)
        ax.imshow(np.reshape(images[idx],(28,28)),cmap='binary')
        title = "label=" +str(np.argmax(labels[idx]))
        if len(prediction)>0:
            title += ",predict="+str(prediction[idx])
            
        ax.set_title(title,fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        idx+=1
    plt.show()


# In[12]:


plot_images_labels_prediction(mnist.test.images,mnist.test.labels,prediction_result,0)


# In[13]:


##View probability
ans = tf.argmax(y_predict, 1)
picture = mnist.test.images[0:1]
with tf.Session() as sess:
    sess.run(init)
    
    saver.restore(sess, model_path)
    print("Restore!")
    
    print("Answer:",sess.run(ans, feed_dict={x: picture}))


# In[14]:


#prediction failure
labels = mnist.test.labels
images = mnist.test.images
test = np.reshape(images[0],(28,28))
plt.imshow(test,cmap='binary')
count =0
predict_fail_index = []
predict_fail_index.clear()
for i in range(10000):
    if np.argmax(labels[i]) != prediction_result[i]:
        count+=1
        predict_fail_index.append(i)
def plot_prediction_failure(images,labels,prediction,idx,num=25):
    fig = plt.gcf()
    fig.set_size_inches(12,14)
    if num>25:
        num=25
    for i in range(0,num):
        number = predict_fail_index[idx]
        ax = plt.subplot(5,5, 1+i)
        ax.imshow(np.reshape(images[number],(28,28)),cmap='binary')
        title = "label=" + str(np.argmax(labels[number]))
        if len(prediction)>0:
            title += ",predict="+str(prediction[number])
            
        ax.set_title(title,fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        idx+=1
    plt.show()


# In[15]:


plot_prediction_failure(mnist.test.images,mnist.test.labels,prediction_result,0)

