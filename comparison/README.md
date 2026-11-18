

\# PathSnap – Machine Learning Model Comparison



\## Overview



This folder documents the comparison and selection of machine learning

models developed for the PathSnap map-matching project.



The objective of P2 is to train and evaluate ML models that assign

probabilities to candidate road segments generated for each GPS

observation.



These candidate probabilities are subsequently used by P3's

HMM/Viterbi decoding pipeline.



\## Models Evaluated



Five models were evaluated:



1\. Logistic Regression

2\. K-Nearest Neighbours (KNN)

3\. Random Forest

4\. LightGBM

5\. Multi-Layer Perceptron (MLP)



Note: The KNN implementation uses class-wise MiniBatchKMeans

prototypes followed by KNN classification. It is an approximate,

prototype-based KNN implementation, not exact KNN over the complete

training dataset.



\## Dataset



Training candidate dataset:



\- File: candidates\_train.csv.gz

\- Candidate rows: 1,407,484

\- GPS observations: 208,309

\- Trips: 80



Training procedure:



\- Unreliable candidate rows excluded.

\- Features kept consistent across all models.

\- GroupKFold cross-validation with 5 folds.

\- Grouping performed using trip\_id.

\- Student-test trips excluded from model tuning.



\## Feature Set



The models use the following 19 features:



\- dist

\- rank

\- dist\_minus\_min

\- dist\_ratio

\- n\_cands

\- proj\_t

\- heading\_diff

\- heading\_diff\_undirected

\- heading\_valid

\- speed

\- dt\_prev

\- dt\_next

\- arc\_length

\- out\_degree

\- in\_degree

\- dist\_prev\_pt

\- dist\_next\_pt

\- in\_prev\_cands

\- in\_next\_cands



Identifiers and target-related columns are excluded from model inputs.



\## Evaluation Metrics



The comparison uses:



\- Candidate Accuracy

\- Macro F1-score

\- Log Loss

\- Observation-level Top-1 Accuracy



Observation-level Top-1 measures whether the highest-probability

candidate selected for each GPS observation is the correct candidate.



This is particularly relevant to the downstream map-matching task.



\## Files



\- model\_comparison.csv: consolidated model metrics

\- MODEL\_COMPARISON\_REPORT.md: detailed methodology and findings

\- MODEL\_SELECTION.md: current model recommendation



\## Important



The reported baseline results are obtained using training-set

cross-validation. They must not be described as final test-set

map-matching accuracy.



The final model must be evaluated separately on the held-out

student-test dataset.



