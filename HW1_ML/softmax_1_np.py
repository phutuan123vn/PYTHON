"""
This file is for fashion mnist classification
"""

import numpy as np
import matplotlib.pyplot as plt
from util import get_mnist_data
from logistic_np import add_one, LogisticClassifier
import time
import pdb

#train_x, train_y, val_x, val_y, test_x, test_y = get_mnist_data()
#print(train_y)

class SoftmaxClassifier(LogisticClassifier):
    def __init__(self, w_shape):
        """__init__
        
        :param w_shape: create w with shape w_shape using normal distribution
        """
        super(SoftmaxClassifier, self).__init__(w_shape)


    def softmax(self, x):
        """softmax
        Compute softmax on the second axis of x
    
        :param x: input
        """
        # [TODO 2.3]
        # Compute softmax

        z = np.matmul(x, self.w)
        z1 = np.zeros((z.shape[0], z.shape[1]), dtype=np.float64)
        zmax = np.amax(z, axis=1)
        for i in range(z.shape[1]):
            z1[:, i] = z[:, i] - zmax
        z1 = np.exp(z1)
        s = np.sum(z1, axis=1)
        for i in range(z.shape[0]):
            z1[i, :] = z1[i, :] / s[i]

        return z1


    def feed_forward(self, x):
        """feed_forward
        This function compute the output of your softmax regression model
        
        :param x: input
        """
        # [TODO 2.3]
        # Compute a feed forward pass



        return self.softmax(x)


    def compute_loss(self, y, y_hat):
        """compute_loss
        Compute the loss using y (label) and y_hat (predicted class)

        :param y:  the label, the actual class of the samples
        :param y_hat: the class probabilities of all samples in our data
        """
        # [TODO 2.4]
        # Compute categorical loss
        loss = 0
        loss = -np.sum(np.sum(np.multiply(y,np.log(y_hat)),axis=1))/y.shape[0]

        return loss
        


    def get_grad(self, x, y, y_hat):
        """get_grad
        Compute and return the gradient of w

        :param loss: computed loss between y_hat and y in the train dataset
        :param y_hat: predicted y
        """ 
        # [TODO 2.5]
        # Compute gradient of the loss function with respect to w
        grad_w = (1/y.shape[0])*np.matmul(x.T,(y_hat-y))


        return grad_w


def plot_loss(train_loss, val_loss):
    plt.figure(1)
    plt.clf()
    plt.plot(train_loss, color='b')
    plt.plot(val_loss, color='g')


def draw_weight(w):
    label_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    plt.figure(2, figsize=(8, 6))
    plt.clf()
    w = w[0:(28*28),:].reshape(28, 28, 10)
    for i in range(10):
        ax = plt.subplot(3, 4, i+1)
        plt.imshow(w[:,:,i], interpolation='nearest')
        plt.axis('off')
        ax.set_title(label_names[i])


def normalize(train_x, val_x, test_x):
    """normalize
    This function computes train mean and standard deviation on all pixels then applying data scaling on train_x, val_x and test_x using these computed values
    Note that in this classification problem, the data is already flatten into a shape of (num_samples, image_width*image_height)

    :param train_x: train images, shape=(num_train, image_height*image_width)
    :param val_x: validation images, shape=(num_val, image_height*image_width)
    :param test_x: test images, shape=(num_test, image_height*image_width)
    """
    # [TODO 2.1]
    # train_mean and train_std should have the shape of (1, 1)

    train_mean = np.mean(train_x)
    train_std = np.mean(train_x)
    train_x = (train_x-train_mean)/train_std
    test_x = (test_x-train_mean)/train_std
    val_x = (val_x-train_mean)/train_std

    return train_x,val_x,test_x

def create_one_hot(labels, num_k=10):
    """create_one_hot
    This function creates a one-hot (one-of-k) matrix based on the given labels

    :param labels: list of labels, each label is one of 0, 1, 2,... , num_k - 1
    :param num_k: number of classes we want to classify
    """
    # [TODO 2.2]
    # Create the one-hot label matrix here based on labels

    m = labels.shape[0]
    one_hot_labels = np.zeros((m,num_k),dtype=np.float64)
    for i in range(labels.shape[0]):
        if labels[i] == 0:
            one_hot_labels[i][0] = 1
        elif labels[i] == 1:
            one_hot_labels[i][1] = 1
        elif labels[i] == 2:
            one_hot_labels[i][2] = 1
        elif labels[i] == 3:
            one_hot_labels[i][3] = 1
        elif labels[i] == 4:
            one_hot_labels[i][4] = 1
        elif labels[i] == 5:
            one_hot_labels[i][5] = 1
        elif labels[i] == 6:
            one_hot_labels[i][6] = 1
        elif labels[i] == 7:
            one_hot_labels[i][7] = 1
        elif labels[i] == 8:
            one_hot_labels[i][8] = 1
        elif labels[i] == 9:
            one_hot_labels[i][9] = 1

    return one_hot_labels

def chuyendoi(x):
    """
    Chuyển đổi y từ dạng one_hot_coding sang labels từ 0-9
    :param x: output one_hot_coding y truyền vào
    :return: trả về labels
    """
    y = np.zeros((x.shape[0],1),dtype=np.float64)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i][j] == 1:
                y[i][0] = j

    return y

def test(y_hat, test_y):
    """test
    Compute the confusion matrix based on labels and predicted values 

    :param classifier: the trained classifier
    :param y_hat: predicted probabilites, output of classifier.feed_forward
    :param test_y: test labels
    """
    
    confusion_mat = np.zeros((10,10))

    
    # [TODO 2.7]
    # Compute the confusion matrix here
    for i in range(test_y.shape[0]):
        confusion_mat[int(test_y[i]),int(y_hat[i])] = confusion_mat[int(test_y[i]),int(y_hat[i])] + 1
    confusion_mat = confusion_mat/np.sum(confusion_mat,axis=1)
    np.set_printoptions(precision=2)
    print('Confusion matrix:')
    print(confusion_mat)
    print('Diagonal values:')
    print(confusion_mat.flatten()[0::11])


if __name__ == "__main__":
    np.random.seed(2020)

    # Load data from file
    # Make sure that fashion-mnist/*.gz files is in data/
    train_x, train_y, val_x, val_y, test_x, test_y = get_mnist_data()
    num_train = train_x.shape[0]
    num_val = val_x.shape[0]
    num_test = test_x.shape[0]  

    #generate_unit_testcase(train_x.copy(), train_y.copy()) 

    # Convert label lists to one-hot (one-of-k) encoding
    train_y = create_one_hot(train_y)
    val_y = create_one_hot(val_y)
    test_y = create_one_hot(test_y)

    # Normalize our data
    train_x, val_x, test_x = normalize(train_x, val_x, test_x)
    
    # Pad 1 as the last feature of train_x and test_x
    train_x = add_one(train_x) 
    val_x = add_one(val_x)
    test_x = add_one(test_x)
    
    # Create classifier
    num_feature = train_x.shape[1]
    dec_classifier = SoftmaxClassifier((num_feature, 10))
    momentum = np.zeros_like(dec_classifier.w)

    # Define hyper-parameters and train-related parameters
    num_epoch = 5000
    learning_rate = 0.01
    momentum_rate = 0.9
    epochs_to_draw = 10
    all_train_loss = []
    all_val_loss = []
    plt.ion()

    for e in range(num_epoch):    
        tic = time.process_time()
        train_y_hat = dec_classifier.feed_forward(train_x)
        val_y_hat = dec_classifier.feed_forward(val_x)

        train_loss = dec_classifier.compute_loss(train_y, train_y_hat)
        val_loss = dec_classifier.compute_loss(val_y, val_y_hat)

        grad = dec_classifier.get_grad(train_x, train_y, train_y_hat)
       
        # dec_classifier.numerical_check(train_x, train_y, grad)
        # Updating weight: choose either normal SGD or SGD with momentum
        dec_classifier.update_weight(grad, learning_rate)
        #dec_classifier.update_weight_momentum(grad, learning_rate, momentum, momentum_rate)


        all_train_loss.append(train_loss) 
        all_val_loss.append(val_loss)
        if (e % epochs_to_draw == epochs_to_draw-1):
            plot_loss(all_train_loss,all_val_loss)
            plt.show()
            plt.pause(0.1)
            print("Epoch %d: train_loss is %.5f val_loss is %.5f" % (e+1, train_loss,val_loss))

        toc = time.process_time()

        #print(toc-tic)
        # [TODO 2.6]
        # Propose your own stopping condition here

    y_hat = dec_classifier.feed_forward(test_x)
    '''for i in range(y_hat.shape[0]):
        print(max(y_hat[i]))'''
    y_hat = np.around(y_hat)
    y_hat = chuyendoi(y_hat)
    test_y = chuyendoi(test_y)
    test(y_hat, test_y)
    print("train_loss",all_train_loss[-1])
    print("val_loss",all_val_loss[-1])