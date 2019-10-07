# Classifiying the Iris Dataset using Multinomial Logistic Regression

Vivek Gopalakrishnan | October 6, 2019


## Problem

The Iris dataset contains four phenotypic measurements from three different species of iris. The challenge is to predict the species of flower from the four measurements.

Specifically, we are given $X_i \in \mathcal{X} \subset \mathbb{R}^4$ and $Y_i \in \mathcal{Y} = \{0, 1, 2\}$. The total number of samples is $n=150$ and the classes are balanced. For prediction, I will use a multonmial logistic regression (a multiclass extension of binary logistic regression). 


## Data

The Iris dataset can be well visualized using a pairs plot. _Setosa_ is completely seperated from the other species, while _Versicolor_ and _Virginica_ show more overlap.

![pairs plot][pairs]


## Model fitting

### Fitting and interpretation
I fit a multinomial logistic regression model. The weights of the model were

```
[[ 0.2660623   0.68677395 -1.13992726 -0.48470934]
 [ 0.11429764 -0.0727098   0.07666636 -0.11054872]
 [-0.38035994 -0.61406414  1.0632609   0.59525806]]
```

and the intercepts were `[ 0.72395208  0.19623416 -0.92018624]`.

Denote the weights corresponding to the ith class (the ith row of the weight matrix) as $\beta^{(i)}$ and the corresponding intercept $b_i$. Then $\Pr{(y=i)} = \exp(\beta^{(i)} \cdot x + b_i) / Z$ where $Z = \sum_i \Pr(y=i)$ is a scaling constant that makes all probabilities sum to 1. Then the exponentiated coefficient of the jth feature from the ith class, $e^{\beta_j^{(i)}}$ is proportional to the amount by which $\Pr(y=i)$ is multiplied by for a unit increase in the feature $X_j$.

### Overfitting
To avoid **overfitting**, I did the following:
- Used a regularized loss function that applied an L2-penalty to the coefficients.
- Limited the amount of training data by training the model on only 10% of the total data set.

To demonstrate that I successfully avoided **overfitting**, I did the following:
- Plotted the decision boundary learned by my classifer and visually confirm that it is not overly complex
- Reported classification accuracy of the testing set in the Performance section.

Below is the decision boundary learned for the features `sepal width` vs `petal length`. It does not suggest that overfitting has occured.

![decision][boundary]

### Performance

I evaluated my model by reporting classification accuracy on the testing set and by reporting a modified Area Under the ROC Curve (AUROC) metric.

- Recall that the model was trained on 10% of all available data. Testing classification accuracy: `0.956`

Because this problem has more than 2 classes, traditional ROC methods do not apply. Instead, I fit three seperate ROC curves (_Setosa_ vs others, _Versicolor_ vs others, and _Virginica_ vs others) and measured the AUROC for these 3 curves.

| Species | AUROC |
| ------- | ----- | 
| _Setosa_  | 1.000 |
| _Versicolor_ | 0.969 |
| _Virginica_ | 0.941 |


## Interpretation

The Iris Dataset is easily classified, but severly limiting the size of the training set can reveal interesting phenomena in the classifier that is used. From the pairs plot above, _Setosa_ is clearly seperated from the other two flowers while _Versicolor_ and _Virginica_ have slight overlap. Recalling that the AUROC is a measure of seperability between two distributions, we can see this fact reflected in the empirical AUROC scores recorded above. _Setosa_ is clearly seperable, so its AUROC is `1.00` while the two classes with more overlap have an average AUROC of `0.95`. These values are still very high, and along with the testing accuracy of `0.956`, demonstrate that a multinomial logistic classifier is effective on the Iris data set even when only trained on 10 of the data.


[pairs]: ./pairs.png 
[boundary]: ./decision.png