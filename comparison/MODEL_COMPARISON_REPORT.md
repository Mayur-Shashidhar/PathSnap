

\# PathSnap – ML Model Comparison Report



\## 1. Objective



The objective of this experiment is to identify a suitable machine

learning classifier for candidate road-segment scoring in PathSnap.



For every GPS observation, the candidate generation module produces

multiple possible road segments.



The ML model estimates the probability that each candidate is the

correct road segment.



These probabilities are intended for use by the HMM/Viterbi

map-matching module.



\## 2. Experimental Setup



All five models use the same 19 candidate-level features.



Only reliable training candidates are used.



Validation uses five-fold GroupKFold cross-validation, with trip\_id

as the grouping variable. This prevents observations from the same

trip appearing in both training and validation folds.



The candidate rows are not treated as independent trips.



\## 3. Models



\### 3.1 Logistic Regression



A linear classification baseline using StandardScaler and

LogisticRegression.



Configuration:

\- C = 1.0

\- Class weight = balanced

\- Solver = lbfgs

\- Maximum iterations = 500



\### 3.2 K-Nearest Neighbours



A prototype-based approximate KNN approach.



MiniBatchKMeans generates class-wise prototypes, followed by

distance-weighted KNN classification.



Configuration:

\- 3,000 prototypes per class

\- 15 nearest neighbours

\- Distance weighting

\- Euclidean distance



This is not exact KNN over all training samples.



\### 3.3 Random Forest



An ensemble of decision trees.



Configuration:

\- 100 estimators

\- Maximum depth = 18

\- Minimum samples per leaf = 2

\- Class weight = balanced\_subsample



\### 3.4 LightGBM



A gradient-boosted decision tree model.



Baseline configuration:

\- 500 estimators

\- Learning rate = 0.05

\- Number of leaves = 31

\- Minimum child samples = 30

\- Subsample = 0.8

\- Feature fraction = 0.8

\- L2 regularisation = 1



\### 3.5 Multi-Layer Perceptron



A feed-forward neural network with two hidden layers.



Configuration:

\- Hidden layers: 64 and 32 neurons

\- ReLU activation

\- Adam optimiser

\- Batch size = 2048

\- Maximum iterations = 30

\- Early stopping enabled



\## 4. Baseline Results



| Model | Candidate Accuracy | Macro F1 | Log Loss | Observation Top-1 |

|---|---:|---:|---:|---:|

| Logistic Regression | 89.06% | 82.45% | 0.2494 | 82.92% |

| Prototype KNN | 88.30% | 81.73% | 0.3051 | 82.01% |

| Random Forest | 94.39% | 89.79% | 0.1379 | 83.88% |

| LightGBM | 95.30% | 90.67% | 0.1155 | 84.27% |

| MLP | See CSV | See CSV | See CSV | 83.53% |



The MLP's exact candidate metrics and log loss should be taken from

the consolidated model\_comparison.csv rather than estimated values.



\## 5. Interpretation



\### Logistic Regression



Provides a useful linear baseline. Its performance is lower than

the tree-based models, suggesting that the relationship between

candidate features and correct road-segment selection is not fully

captured by a linear decision boundary.



\### Prototype KNN



Provides a non-parametric comparison but has lower performance

than the other evaluated approaches.



Its prototype-based approximation reduces computational cost

compared with searching the complete training set.



\### Random Forest



Shows a substantial improvement over the linear and KNN baselines.



It captures non-linear feature relationships and interactions.



\### LightGBM



LightGBM achieves the strongest reported baseline results:



\- Candidate accuracy: 95.30%

\- Macro F1: 90.67%

\- Log loss: 0.1155

\- Observation Top-1: 84.27%



It also has relatively fast training and inference compared with

the Random Forest baseline.



\### MLP



The MLP achieves an observation-level Top-1 score of 83.53%.

Its performance is competitive, but it does not exceed the

LightGBM baseline on this metric.



\## 6. Model Selection



LightGBM is selected as the current leading baseline candidate.



The selection is based on its candidate-level classification

performance, probability quality as measured by log loss, and

observation-level Top-1 score.



However, final selection is pending:

\- Optuna hyperparameter tuning

\- Fair five-fold comparison of tuned and baseline LightGBM

\- Held-out student-test evaluation

\- Integration with the HMM/Viterbi decoder



\## 7. Limitations



1\. Candidate classification accuracy is not equivalent to complete

&#x20;  route-level map-matching accuracy.

2\. Some GPS observations do not contain a correct candidate in

&#x20;  the generated candidate set.

3\. The KNN model is prototype-based and approximate.

4\. The baseline results are cross-validation results, not

&#x20;  held-out test results.

5\. The HMM/Viterbi decoder may improve route consistency, but

&#x20;  its contribution must be evaluated separately.



\## 8. Conclusion



LightGBM is the strongest baseline candidate among the evaluated

models based on the currently available metrics.



Its candidate probabilities will be used as the initial input

for downstream map-matching integration, subject to final

validation and interface compatibility.



