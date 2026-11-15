# Reports

This folder is reserved for the final written report and supporting figures or tables.
Generated intermediate outputs remain in notebooks or temporary directories.

## Draft: model comparison

This section is a draft summary copied from
[`notebooks/model_comparison.ipynb`](../notebooks/model_comparison.ipynb). The models
were compared on the chronological validation partition. Macro-F1 was the primary
selection metric because accuracy can be high for a model that mainly predicts the
most common class.

| Model | Validation macro-F1 | Validation accuracy |
|---|---:|---:|
| LightGBM | 0.3265 | 0.3944 |
| Logistic regression | 0.3093 | 0.3753 |
| Majority baseline | 0.1213 | 0.4354 |

The majority baseline has the highest accuracy, but it predicts `on_time` for every
row and has zero recall for the other four classes. Logistic regression identifies all
five classes and substantially improves macro-F1. LightGBM is the best overall model
because it achieves the highest validation macro-F1 and handles nonlinear
relationships and interactions between categorical and temporal features more
effectively than logistic regression.

The train-type confusion matrices show that performance is not uniform across
services. IC and ICE have a strong LightGBM prediction bias toward
`severe_or_cancelled`, while RE is more balanced but still tends toward `on_time`.
This suggests that combining all three train types in one model may be suboptimal
because their class distributions and feature relationships differ. The five-class
target is also difficult because neighboring delay categories overlap substantially;
a three-class target may be easier to distinguish and more useful.

LightGBM is therefore the selected model for this comparison based on validation
macro-F1. The final test evaluation is kept separate from model selection.

### Final test evaluation

The chronological test partition contains 2,126,112 rows and was not used for
training, tuning, or model selection. After LightGBM was selected using validation
macro-F1, it was evaluated once on this untouched partition. The test features were
transformed using preprocessing rules fitted on the training partition.

| Model | Test rows | Test macro-F1 | Test accuracy |
|---|---:|---:|---:|
| LightGBM | 2,126,112 | 0.3228 | 0.3962 |

The test macro-F1 is close to the validation result, suggesting that the selected
LightGBM model maintains similar performance on later unseen data. These test results
are reported only as a final estimate and were not used to change the model.

### Validation results by train type

| Train type | Model | Validation rows | Macro-F1 | Accuracy |
|---|---|---:|---:|---:|
| IC | LightGBM | 68,991 | 0.2616 | 0.3148 |
| IC | Logistic regression | 68,991 | 0.2057 | 0.2673 |
| IC | Majority baseline | 68,991 | 0.0946 | 0.3096 |
| ICE | LightGBM | 234,741 | 0.2233 | 0.2966 |
| ICE | Logistic regression | 234,741 | 0.1460 | 0.2453 |
| ICE | Majority baseline | 234,741 | 0.0846 | 0.2683 |
| RE | LightGBM | 1,816,624 | 0.3136 | 0.4101 |
| RE | Logistic regression | 1,816,624 | 0.2989 | 0.3961 |
| RE | Majority baseline | 1,816,624 | 0.1264 | 0.4618 |

LightGBM has the highest macro-F1 for each train type. RE performs best overall,
while ICE is the most difficult subgroup. The majority baseline has higher accuracy
for RE because `on_time` is common, but its much lower macro-F1 shows that it does
not recognize the other classes.

### LightGBM confusion matrices by train type

The following row-normalized matrices show percentages of each true class predicted
as each target class. Rows are true labels and columns are predicted labels, in the
order `on_time`, `slight_delay`, `late`, `delayed`, `severe_or_cancelled`.

**IC**

| True class | on_time | slight_delay | late | delayed | severe_or_cancelled |
|---|---:|---:|---:|---:|---:|
| on_time | 30.8% | 5.0% | 3.5% | 6.5% | 54.3% |
| slight_delay | 12.2% | 7.8% | 10.9% | 7.8% | 61.3% |
| late | 6.8% | 4.2% | 16.7% | 10.9% | 61.5% |
| delayed | 4.3% | 2.2% | 7.6% | 11.7% | 74.1% |
| severe_or_cancelled | 3.2% | 1.4% | 4.8% | 8.0% | 82.5% |

**ICE**

| True class | on_time | slight_delay | late | delayed | severe_or_cancelled |
|---|---:|---:|---:|---:|---:|
| on_time | 27.4% | 3.8% | 1.8% | 10.5% | 56.4% |
| slight_delay | 12.7% | 5.8% | 3.3% | 13.2% | 65.0% |
| late | 7.2% | 2.9% | 3.1% | 16.2% | 70.7% |
| delayed | 4.9% | 1.5% | 1.7% | 16.1% | 75.9% |
| severe_or_cancelled | 2.5% | 0.9% | 0.9% | 11.2% | 84.5% |

**RE**

| True class | on_time | slight_delay | late | delayed | severe_or_cancelled |
|---|---:|---:|---:|---:|---:|
| on_time | 57.5% | 18.3% | 8.8% | 8.2% | 7.2% |
| slight_delay | 33.4% | 29.9% | 16.7% | 12.0% | 8.0% |
| late | 23.4% | 23.3% | 24.5% | 19.4% | 9.4% |
| delayed | 19.7% | 16.8% | 21.1% | 30.6% | 11.8% |
| severe_or_cancelled | 24.0% | 17.5% | 15.6% | 23.1% | 19.8% |

These matrices explain the subgroup results: IC and ICE have a strong LightGBM bias
toward `severe_or_cancelled`, whereas RE is more balanced but leans toward
`on_time`. The complete comparison notebook also contains matrices for logistic
regression and the majority baseline.