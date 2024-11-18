# music-genre-classification-using-lyrics
## Problem Setting
In the realm of music analysis, genre classification plays a crucial role in various applications, including
recommendation systems, music streaming services, and music information retrieval. By accurately
categorizing songs into different genres based on their musical features and characteristics, these systems
can provide personalized recommendations to users, enhance music discovery experiences, and facilitate
music organization and search functionalities.
Traditionally, genre classification has primarily relied on audio features such as tempo, rhythm, pitch, and
timbre. However, lyrics also contain valuable information that can significantly contribute to genre
classification tasks. The linguistic content of lyrics, including vocabulary, themes, and sentiment, can
convey important cues about the underlying genre of a song. For example, country songs often feature
themes of rural life, love, and heartbreak, while rap music may focus on urban culture, social issues, and
personal narratives.

## Data Mining Models/Methods
Baseline Models:
For the initial phase of genre classification based on lyrics, three baseline models were employed:
Random Forest, Logistic Regression, and XGBoost. These models were chosen due to their versatility and
effectiveness in handling classification tasks.
Logistic Regression:
- Logistic Regression is a statistical model that is used to model the probability of a certain class or
event occurring based on the input features. Despite its name, it is primarily used for binary classification
tasks, but can be extended to multi-class classification through techniques like one-vs-rest or softmax.
Fine-Tuned Model:
To enhance the performance of genre classification, a fine-tuned model was developed by refining the
parameters of the XGBoost and Random Forest algorithm through hyperparameter tuning. The following
hyperparameters were optimized:
XGBoost Parameters
eta (Learning Rate): Set to 0.1 to control the step size shrinkage.
n_estimators: Number of boosting rounds (trees) set to 100.
max_depth: Maximum depth of a tree set to 6 to control overfitting.
reg_lambda: L2 regularization term on weights, set to 1.
reg_alpha: L1 regularization term on weights, set to 0.1.
gamma: Minimum loss reduction required to make a further partition on a leaf node of the tree, set to 0.1.
subsample: Subsample ratio of the training instances, set to 0.8.
colsample_bytree: Subsample ratio of columns when constructing each tree, set to 0.8.
colsample_bylevel: Subsample ratio of columns for each level, set to 0.8.
colsample_bynode: Subsample ratio of columns for each node, set to 0.8.
tree_method: Tree construction algorithm used, set to 'hist' for faster performance.
verbosity: Verbosity of the XGBoost model, set to 1 for printing messages during training.
Random Forest:
n_estimators: This parameter specifies the number of trees in the forest. In your model, you’ve set it to
100, which means that 100 different decision trees will be constructed in the ensemble. The more trees,
the more robust the model, but also the greater the computational cost.
max_depth: This controls the maximum depth of each tree. A depth of 20 allows the trees to grow until
they contain 20 splits or until all leaves are pure or contain less than min_samples_split samples. If not
set, the nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split
samples.
min_samples_split: The minimum number of samples required to split an internal node. A value of 2
means that at least two samples are required to justify creating a new branch in a tree. Higher values
prevent creating splits that do not provide enough statistical power.
min_samples_leaf: The minimum number of samples required to be at a leaf node. A leaf is the end of a
branch in a decision tree. A minimum of 1 means that each leaf can have just one sample, which can
make the model more prone to capturing noise in training data.
max_features: The number of features to consider when looking for the best split. ‘sqrt’ means that only
a subset of the features are considered, specifically the square root of the total number of features. This
adds randomness to the model and can help improve generalization.
criterion: The function used to measure the quality of a split. ‘gini’ refers to Gini impurity, a measure of
how often a randomly chosen element from the set would be incorrectly labeled if it was randomly
labeled according to the distribution of labels in the subset.
class_weight: This parameter is used to weigh the classes if the classes are imbalanced. ‘None’ means all
classes are supposed to have weight one. ‘balanced’ automatically adjusts weights inversely proportional
to class frequencies in the input data.
oob_score: Stands for Out-of-Bag score, which is a method of measuring the prediction error of random
forests and other bagging classifiers. When set to ‘False’, it means this method is not used. When ‘True’,
it uses out-of-bag samples to estimate the generalization accuracy.
random_state: Controls both the randomness of the bootstrapping of the samples used when building
trees (if bootstrap=True) and the sampling of the features to consider when looking for the best split at
each node. The value of 42 ensures that the same sequence of random numbers is generated each time you
run the code, which is useful for reproducibility.
verbose: Controls the verbosity when fitting and predicting. A value of 0 means that no output is
generated during the training process.
These parameters collectively determine how the RandomForestClassifier will be constructed, trained,
and make predictions. Fine-tuning these parameters can significantly impact the performance of the
model.
