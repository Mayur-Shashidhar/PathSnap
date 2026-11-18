

\# PathSnap – Current Model Selection



\## Selected Baseline: LightGBM



LightGBM is the current preferred baseline model for P3 integration.



\### Why LightGBM?



| Metric | Baseline Result |

|---|---:|

| Candidate Accuracy | 95.30% |

| Macro F1 | 90.67% |

| Log Loss | 0.1155 |

| Observation Top-1 | 84.27% |



\### Why Observation Top-1 Matters



Each GPS observation has multiple candidate road segments.



The classifier assigns a probability to each candidate. The

highest-probability candidate is selected for the observation.



Observation Top-1 measures how often that candidate is correct.



This is more directly relevant to candidate selection than

candidate-row accuracy alone.



\### P3 Integration



P3 should receive candidate probabilities associated with:



\- trip\_id

\- point\_idx

\- arc\_id

\- probability



The HMM/Viterbi decoder can then combine candidate scores with

transition and connectivity information.



\### Final Selection Status



Current status: Baseline selected provisionally.



Final selection requires:

1\. Optuna tuning.

2\. Five-fold validation using identical splits.

3\. Held-out student-test evaluation.

4\. End-to-end HMM/Viterbi evaluation.



