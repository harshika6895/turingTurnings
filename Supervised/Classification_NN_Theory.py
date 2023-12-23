#!/usr/bin/env python
# coding: utf-8

# WHAT MAKES UP A CLASSIFICATION MODEL? 
# 
# The classification model break up into 3 parts:
# 
# 1)THE STRUCTURE OF THE MODEL : We use a threshold on a single feature
# 
# 2)THE SEARCH PROCEDURE : We tru every possible combination of feature and threshold 
# 
# 3)THE LOSS FUNCTION: Using the loss function we decide which of the possibilities is less bad. We want the loss function to be minimum.
# 
# Playing around with these 3 parts gives us the best model to work with. We can attempt to build the threshold with minimum training error, but we will only test 3 values for every feature which are mean value of the feature, the mean plus one standard deviation and the mean minus one standard deviation.
# Alternatively we can also have different loss functions because it might be the case that one type of error is more costly than another. For eg. in medical settings false positives and false negatives are not equivalent.Same with spam filterling, deleting a non-spam-email which is false negative error is much dangerous for the user that getting a spam mail through which is false positive and is not that big of a deal.
# 
# A MORE COMPLEX DATASET - SEEDS DATASET 
# 
# In this dataset we have data about the seed of 3 species of wheat. There are 7 features given : Area , Perimeter , Compactness of the kernel(C = 4piA/P^2) , Length of kernel , width of kernel, Asymmetry coefficient, Length of kernel groove. The Three classes of wheat are CANADIAN, KOMA and ROSA.
# 
# FEATURES AND FEATURE ENGINEERING : Feature engineering is not so fancy but is important. In this we derive a new combined feature from our given features increasing the performance. For eg in the seed dataset compactness is a derived feature from area and perimeter is a typical feature for shapes. This feature will have the same value for two kernels, oone of which is twice as big as the other one but with the same shape. However they will have different values for kernels with different shapes like for round and elongated one.
# 
# This feature engineering is done to get a good feature. The good feature is the once which simultaneously varies with what matters and be invariant with what doesnt. To know the good feature you must have some background knowldege to intuit which will be the good feature. There are several problem domains where there is already a vast literature for possible features and types that we can build upon, but you can always use your knowledge of the specific problem to design a specific feature. "Even Before you have data, you must decide which data is worthwhile to collect.
# 
# FEATURE SELECTION : The problem that whether we can select the good feature automatically. There are many methods which work for the problem but very simple ideas works the best.
# 
# NEAREST NEIGHBOR CLASSIFICATION :
# 
# Consider representing each example of the datasert as a point in N-dimensional space by its features, calculating the distance between its neighbour points and getting the minimum distanced point's label as your own label. This is what we call Nearest Neighbour classification. 
# 
# Defining a function to calculate the distance between 2points as shown : 

# In[ ]:


def distance(p0 , p1):
    return np.sum((p0-p1)**2)


# Define a function to find the nearest neighbour and predictig our points label as the nearest neighbours label:

# In[ ]:


def nn_classify(training_set, training_lables , new_example):
    dist = np.array([distance(t , new_example) 
                     for t in training_set])
    nearest = dist.argmin()
    return training_labels[nearest]


# NOTE: The models performs perfectly on the training data as every point of training data is the nearest neighbour to itself giving the prediction of its label 100% right, only if there is no other point with same features and different label which is possible therefore it is necessary to test using a CROSS VALIDATION PROTOCOL.
# 
# CROSS VALIDATION accuracy is lower than training accuracy but is more credible estimate of the performance of the model.
# 
# NOW examining the DECISION BOUNDARY. Let's look only 2 dimensions.For our seed daataset there are 7 features giving the point in a 7-D space and then finding the distance in order to predict bu NN classifier. But the issue here is that all the 7 features have different units and in order to make a 7d space we have to normalize all the features in same dimensions so that the distance is a true value to compare considering all features. 
# 
# Z-SCORES - In order to normalize we have several solutions and Z scores is one of them. The Z-Score of a value is how far away from the mean it is in terms of units of standard deviation. operations:

# In[ ]:


features -=features.mean(axis=0)
features /=features.std(axis=0)


# Values of features after Zscoring are 0 is the mean and above it are the positive values and below it are the negative values. After Z-scoring every feature has trhe same dimension(dimensionless). NOw the NN classifier has more accuracy.
# 
# We can generalize NN classifier to K-NN classifier by considering K nearest neighbours and all k nearest neighbours will vote to select the label. 
# 
# BINARY and MULTICLASS CLASSIFIER: The two classifiers - threshold classifier and NN classifier are Binary classifier and Multiclass Classifier respectively. Here We can understand that every multiclass dataset can be classified using the binary classifier. We just have to use the right binary classifier at each level and perform the classification until we get our class individually. This can easily be seen with a classification tree. 
# 
# 
# REFERENCE : 
# BUILDING MACHINE LEARNING SUSTEMS WITH PYTHON : willi richert and luis pedro coelho Chapter 2: Learning how to classify with real world examples.
# 
