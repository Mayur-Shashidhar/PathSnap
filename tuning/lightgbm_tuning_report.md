\# LightGBM Hyperparameter Tuning – PathSnap P2



\## 1. Objective

Optimise the LightGBM classifier's hyperparameters to improve candidate road ranking performance.



\## 2. Optimisation Framework

\- Framework: Optuna

\- Sampler: TPE (default TPESampler)

\- Optimisation method: Bayesian optimisation

\- Number of trials: 15

\- Validation: 3-fold GroupKFold

\- Grouping variable: trip\_id

\- Optimisation objective: Maximise observation-level Top-1 accuracy



\## 3. Best Trial

\- Best trial: Trial 4

\- Best 3-fold validation Top-1: 83.76%



\### Best Hyperparameters



| Hyperparameter | Value |

|---|---:|

| n\_estimators | 300 |

| learning\_rate | 0.03178697 |

| num\_leaves | 35 |

| max\_depth | 8 |

| min\_child\_samples | 69 |

| subsample | 0.95924959 |

| colsample\_bytree | 0.86701989 |

| reg\_alpha | 0.69403430 |

| reg\_lambda | 2.57570630 |



\## 4. Baseline vs Tuned Model



Both models were evaluated using the same 5-fold GroupKFold setup.



| Metric | Baseline | Tuned |

|---|---:|---:|

| Candidate Accuracy | 95.303% | 95.301% |

| Macro F1 | 90.666% | 90.677% |

| Log Loss | 0.115540 | 0.115846 |

| Observation Top-1 | 84.274% | 84.156% |

| Mean Training Time per Fold | 21.84 s | 15.39 s |



\## 5. Conclusion



Hyperparameter tuning did not produce a meaningful overall improvement over the baseline LightGBM model.



The baseline achieved better candidate accuracy, log loss and observation-level Top-1 accuracy. The tuned model had marginally higher Macro F1 and lower training time.



Therefore, the original baseline LightGBM model was retained as the selected P2 model.



\## 6. Important Notes

\- The 83.76% tuning score comes from 3-fold validation.

\- The baseline-versus-tuned comparison uses the same 5-fold validation setup.

\- The student test dataset was not used for hyperparameter optimisation.

\- TPE uses previous trial results to guide subsequent hyperparameter suggestions.

