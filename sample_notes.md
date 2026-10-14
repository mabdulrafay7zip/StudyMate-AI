# Introduction to Machine Learning — Study Notes

## What is Machine Learning?

Machine learning is a branch of artificial intelligence in which computers learn patterns from data instead of being explicitly programmed with rules. A machine learning system improves its performance at a task as it is exposed to more examples of that task. For example, an email spam filter learns from thousands of labelled emails which messages look like spam, rather than following a hand-written list of banned words. The key ingredients of any machine learning project are data, a model that can represent patterns in that data, and a way to measure how well the model is doing.

## Types of Machine Learning

There are three main types of machine learning. In supervised learning, the training data comes with labels, so the model learns to map inputs to known outputs, such as predicting a house price or classifying an image. In unsupervised learning, the data has no labels, and the model must discover structure on its own, for example by grouping customers into clusters with similar behaviour. In reinforcement learning, an agent learns by interacting with an environment, taking actions and receiving rewards or penalties, gradually discovering which actions lead to the best long-term outcome. Most introductory courses focus mainly on supervised learning because its results are easiest to measure.

## Classification and Regression

Supervised learning problems are usually divided into classification and regression. Classification predicts a discrete category, such as whether a tumour is benign or malignant, or which digit appears in an image. Regression predicts a continuous number, such as the temperature tomorrow or the price of a used car. Choosing the wrong framing is a common beginner mistake: if the output is a category, it is classification even when the categories happen to be numbered. Many algorithms exist in both flavours, for example decision trees can be grown as classifiers or as regressors.

## Training, Validation and Test Data

Before training, a dataset is normally split into three parts. The training set is used to fit the model's parameters. The validation set is used to tune choices such as hyperparameters and to compare candidate models. The test set is kept completely aside and used only once at the end, to estimate how the model will perform on genuinely unseen data. A typical split might be 70 percent training, 15 percent validation and 15 percent test. Keeping the test set untouched matters because repeatedly peeking at it quietly turns it into training data and gives an over-optimistic estimate of performance.

## Overfitting and Underfitting

Overfitting happens when a model learns the training data too well, memorising noise and random quirks instead of the underlying pattern. An overfitted model scores very high on training data but performs poorly on new data. Underfitting is the opposite problem: the model is too simple to capture the pattern at all, so it performs badly on both training and new data. The goal is a balance between the two, sometimes described as the bias-variance tradeoff. Regularisation, simpler models, more training data and cross-validation are standard tools for fighting overfitting.

## Evaluation Metrics: Accuracy, Precision, Recall and F1

Accuracy is the fraction of all predictions that are correct, and it is simple but misleading when classes are imbalanced. Precision answers the question: of all the items the model labelled positive, how many were actually positive? Recall answers a different question: of all the truly positive items, how many did the model manage to find? The F1 score is the harmonic mean of precision and recall, giving a single number that is high only when both are high. For a cancer screening test, recall is usually the priority, because missing a real case is far more costly than a false alarm.

## Gradient Descent

Gradient descent is the optimisation algorithm used to train most machine learning models. It works by repeatedly taking small steps downhill on a loss function, where downhill means the direction that reduces the model's error. The size of each step is controlled by a parameter called the learning rate. If the learning rate is too large, the algorithm may overshoot the minimum and never settle; if it is too small, training becomes painfully slow. The gradient itself is computed with calculus: it is the derivative of the loss with respect to each model parameter, telling us how a small change in that parameter would change the error.

## Decision Trees

A decision tree classifies data by asking a sequence of yes-or-no questions about the features, splitting the data at each step so that the resulting groups are as pure as possible. The purity of a split can be measured with entropy or the Gini impurity. Trees are popular with beginners because they are easy to visualise and explain to non-experts. However, a single deep tree tends to overfit badly, which is why ensembles such as random forests, which average the votes of many trees trained on different samples of the data, usually perform better.

## Introduction to Neural Networks

A neural network is a model made of layers of simple units called neurons, loosely inspired by the brain. Each neuron computes a weighted sum of its inputs, adds a bias, and passes the result through a non-linear activation function such as ReLU or sigmoid. Neurons are organised into an input layer, one or more hidden layers, and an output layer. Networks with many hidden layers are called deep networks, and training them is known as deep learning. Neural networks learn by backpropagation, which applies the chain rule of calculus to distribute the blame for a prediction error across all the weights in the network.

## Data Preprocessing

Raw data almost never arrives ready for training. Preprocessing includes handling missing values, for example by filling them with the mean of a column or dropping incomplete rows. It also includes encoding categorical variables into numbers, because most algorithms cannot work with text labels directly. Feature scaling, such as standardisation or min-max normalisation, puts features measured on different scales onto a comparable footing, which is essential for distance-based algorithms like k-nearest neighbours. Careful preprocessing often improves results more than switching to a fancier model, and it must be fitted on the training data only, otherwise information from the test set leaks into training.
