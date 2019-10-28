'''https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html'''
'''
In Depth: Principal Component Analysis

Up until now, we have been looking in depth at supervised learning estimators: those estimators that predict labels
based on labeled training data. Here we begin looking at several unsupervised estimators, which can highlight
interesting aspects of the data without reference to any known labels.

In this section, we explore what is perhaps one of the most broadly used of unsupervised algorithms, principal component
 analysis (PCA). PCA is fundamentally a dimensionality reduction algorithm, but it can also be useful as a tool for
 visualization, for noise filtering, for feature extraction and engineering, and much more. After a brief conceptual
 discussion of the PCA algorithm, we will see a couple examples of these further applications.

We begin with the standard imports:'''

# %matplotlib inline
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

'''
Introducing Principal Component Analysis

Principal component analysis is a fast and flexible unsupervised method for dimensionality reduction in data, which we
saw briefly in Introducing Scikit-Learn. Its behavior is easiest to visualize by looking at a two-dimensional dataset.
Consider the following 200 points:'''

rng = np.random.RandomState(1)
X = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T
plt.scatter(X[:, 0], X[:, 1])
plt.axis('equal')
plt.show()
plt.clf()

'''By eye, it is clear that there is a nearly linear relationship between the x and y variables. This is reminiscent of
the linear regression data we explored in In Depth: Linear Regression, but the problem setting here is slightly
different: rather than attempting to predict the y values from the x values, the unsupervised learning problem attempts
to learn about the relationship between the x and y values.

In principal component analysis, this relationship is quantified by finding a list of the principal axes in the data,
and using those axes to describe the dataset. Using Scikit-Learn's PCA estimator, we can compute this as follows:'''

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X)

'''The fit learns some quantities from the data, most importantly the "components" and "explained variance":'''

print(pca.components_)
print '\n'
print(pca.explained_variance_)

'''To see what these numbers mean, let's visualize them as vectors over the input data, using the "components" to define
 the direction of the vector, and the "explained variance" to define the squared-length of the vector:'''

def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth=2,
                    shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)

# plot data
plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
for length, vector in zip(pca.explained_variance_, pca.components_):
    v = vector * 3 * np.sqrt(length)
    draw_vector(pca.mean_, pca.mean_ + v)
plt.axis('equal')
plt.show()
plt.clf()

'''These vectors represent the principal axes of the data, and the length of the vector is an indication of how
important that axis is in describing the distribution of the data. More precisely, it is a measure of the variance
of the data when projected onto that axis. The projection of each data point onto the principal axes are the "principal
components" of the data.'''

'''This transformation from data axes to principal axes is an affine transformation, which basically means it is
composed of a translation, rotation, and uniform scaling.

While this algorithm to find principal components may seem like just a mathematical curiosity, it turns out to have
very far-reaching applications in the world of machine learning and data exploration.'''


'''
PCA as dimensionality reduction

Using PCA for dimensionality reduction involves zeroing out one or more of the smallest principal components,
resulting in a lower-dimensional projection of the data that preserves the maximal data variance.

Here is an example of using PCA as a dimensionality reduction transform:'''

pca = PCA(n_components=1)
pca.fit(X)
X_pca = pca.transform(X)
print("original shape:   ", X.shape)
print("transformed shape:", X_pca.shape)

'''The transformed data has been reduced to a single dimension. To understand the effect of this dimensionality
reduction, we can perform the inverse transform of this reduced data and plot it along with the original data:'''

X_new = pca.inverse_transform(X_pca)
plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
plt.scatter(X_new[:, 0], X_new[:, 1], alpha=0.8)
plt.axis('equal')
plt.show()
plt.clf()

'''The light points are the original data, while the dark points are the projected version. This makes clear what a PCA
dimensionality reduction means: the information along the least important principal axis or axes is removed, leaving
only the component(s) of the data with the highest variance. The fraction of variance that is cut out (proportional to
the spread of points about the line formed in this figure) is roughly a measure of how much "information" is discarded
 in this reduction of dimensionality.

This reduced-dimension dataset is in some senses "good enough" to encode the most important relationships between the
points: despite reducing the dimension of the data by 50%, the overall relationship between the data points are mostly
preserved.'''

'''
PCA for visualization: Hand-written digits

The usefulness of the dimensionality reduction may not be entirely apparent in only two dimensions, but becomes much
more clear when looking at high-dimensional data. To see this, let's take a quick look at the application of PCA to the
digits data we saw in In-Depth: Decision Trees and Random Forests.

We start by loading the data:'''

from sklearn.datasets import load_digits
digits = load_digits()
print digits.data.shape

'''Recall that the data consists of 8*8 pixel images, meaning that they are 64-dimensional. To gain some intuition into
 the relationships between these points, we can use PCA to project them to a more manageable number of dimensions, say
 two:'''

pca = PCA(2)  # project from 64 to 2 dimensions
projected = pca.fit_transform(digits.data)
print(digits.data.shape)
print(projected.shape)

'''We can now plot the first two principal components of each point to learn about the data:'''

plt.scatter(projected[:, 0], projected[:, 1],
            c=digits.target, edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('spectral', 10))
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.colorbar()
plt.show()
plt.clf()

'''Recall what these components mean: the full data is a 64-dimensional point cloud, and these points are the projection
 of each data point along the directions with the largest variance. Essentially, we have found the optimal stretch and
 rotation in 64-dimensional space that allows us to see the layout of the digits in two dimensions, and have done this
 in an unsupervised manner that is, without reference to the labels.'''

'''
What do the components mean?
We can go a bit further here, and begin to ask what the reduced dimensions mean. This meaning can be understood in terms
 of combinations of basis vectors. For example, each image in the training set is defined by a collection of 64 pixel
 values, which we will call the vector x
:

x=[x1,x2,x3.....x64]'''

'''One way we can think about this is in terms of a pixel basis. That is, to construct the image, we multiply each
element of the vector by the pixel it describes, and then add the results together to build the image:

image(x)=x1 point (pixel 1)+x2 point (pixel 2)+x3 point (pixel 3)...x64 point (pixel 64)

One way we might imagine reducing the dimension of this data is to zero out all but a few of these basis vectors.
For example, if we use only the first eight pixels, we get an eight-dimensional projection of the data, but it is not
very reflective of the whole image: we've thrown out nearly 90% of the pixels!'''

'''
Choosing the number of components

A vital part of using PCA in practice is the ability to estimate how many components are needed to describe the data.
This can be determined by looking at the cumulative explained variance ratio as a function of the number of components:'''

pca = PCA().fit(digits.data)
# print np.cumsum(pca.explained_variance_ratio_)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()
plt.clf()

'''This curve quantifies how much of the total, 64-dimensional variance is contained within the first N
 components. For example, we see that with the digits the first 10 components contain approximately 75% of the variance,
  while you need around 50 components to describe close to 100% of the variance.

Here we see that our two-dimensional projection loses a lot of information (as measured by the explained variance) and
that we'd need about 20 components to retain 90% of the variance. Looking at this plot for a high-dimensional dataset
can help you understand the level of redundancy present in multiple observations.'''

'''
PCA as Noise Filtering

PCA can also be used as a filtering approach for noisy data. The idea is this: any components with variance much larger
than the effect of the noise should be relatively unaffected by the noise. So if you reconstruct the data using just the
 largest subset of principal components, you should be preferentially keeping the signal and throwing out the noise.

Let's see how this looks with the digits data. First we will plot several of the input noise-free data:'''


def plot_digits(data):
    fig, axes = plt.subplots(4, 10, figsize=(10, 4),
                             subplot_kw={'xticks':[], 'yticks':[]},
                             gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes.flat):
        ax.imshow(data[i].reshape(8, 8),
                  cmap='binary', interpolation='nearest',
                  clim=(0, 16))
plot_digits(digits.data)
plt.show()
plt.clf()

'''Now lets add some random noise to create a noisy dataset, and re-plot it:'''

np.random.seed(42)
noisy = np.random.normal(digits.data, 4)
plot_digits(noisy)
plt.show()
plt.clf()

'''It's clear by eye that the images are noisy, and contain spurious pixels. Let's train a PCA on the noisy data,
requesting that the projection preserve 50% of the variance:'''

pca = PCA(0.50).fit(noisy)
print pca.n_components_

'''Here 50% of the variance amounts to 12 principal components. Now we compute these components, and then use the
inverse of the transform to reconstruct the filtered digits:'''

components = pca.transform(noisy)
filtered = pca.inverse_transform(components)
plot_digits(filtered)
plt.show()
plt.clf()

'''This signal preserving/noise filtering property makes PCA a very useful feature selection routine. for example,
rather than training a classifier on very high-dimensional data, you might instead train the classifier on the
lower-dimensional representation, which will automatically serve to filter out random noise in the inputs.'''