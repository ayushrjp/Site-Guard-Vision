# Class imbalance investigation

## Scope and data

Counts below come from the existing train/validation YOLO label files. No
training data, labels, or sampling strategy was modified. False-negative rates
are measured from the controlled validation predictions at confidence 0.25 and
IoU 0.50, using the same custom class-aware one-to-one matching protocol as the
baseline evaluator.

| Class | Train instances | Validation instances | Train instances per validation instance |
|---|---:|---:|---:|
| Helmet | 2,116 | 232 | 9.12 |
| No Helmet | 94 | 11 | 8.55 |
| No Vest | 741 | 90 | 8.23 |
| Person | 2,362 | 241 | 9.80 |
| Safety Vest | 1,073 | 141 | 7.61 |

The training set contains **22.5×** more Helmet than No Helmet annotations;
No Helmet is 4.4% of the Helmet count. No Vest has **69.1%** of the Safety Vest
count (Safety Vest has 1.45× as many examples). No Helmet is a much stronger
count imbalance than the No Vest/Safety Vest pair.

## Validation false-negative rates

Rates below are object-level FN / ground-truth count at the unchanged 0.25
reference confidence threshold, with a match requiring same class and IoU ≥0.50.
They are descriptive of this small fixed validation set and are not causal
estimates of the effect of class counts.

| Class | Validation GT | FN at 640 | FN rate at 640 | FN at 960 | FN rate at 960 |
|---|---:|---:|---:|---:|---:|
| Helmet | 232 | 40 | 17.2% | 49 | 21.1% |
| No Helmet | 11 | 2 | 18.2% | 8 | 72.7% |
| No Vest | 90 | 28 | 31.1% | 52 | 57.8% |
| Safety Vest | 141 | 25 | 17.7% | 44 | 31.2% |

For context, the 800px false-negative counts/rates were Helmet 42/18.1%,
No Helmet 4/36.4%, No Vest 33/36.7%, and Safety Vest 31/22.0%. The 640px
configuration had lower FN rates than 960px for each of the four PPE classes
on this validation set. This is an observed inference-resolution association,
not proof that resolution caused the difference beyond this dataset.

## Class-aware sampling: potential benefits and risks

**Potential benefit:** oversampling rare No Helmet examples or using a
class-aware sampler could increase the number of minority-class examples seen
per training epoch, potentially improving representation and recall without
collecting new labels. The 94-example No Helmet training class is the clearest
candidate by raw count.

**Risks:** repeated exposure to a very small set can overfit scenes/identities,
increase false positives, harm calibration, and distort the natural class
prior. Images contain multiple labels, so image-level oversampling can also
increase unrelated classes. No Vest is less common than Safety Vest, but its
count gap is modest relative to the Helmet/No Helmet gap; aggressive balancing
there is not justified by counts alone.

## Additional training data: potential benefits and risks

**Potential benefit:** new, diverse examples can expand the appearance and
context coverage of hard classes and reduce repetition. Collection should
target confirmed failure conditions and include both positive and negative
examples under the annotation policy.

**Risks:** annotation disagreement can add noise; collecting only obvious
minority examples can create sampling bias; more data costs review/training
time and does not guarantee better recall. The two flagged test images remain
review-only and must not be copied into training or validation.

## Conclusion

Counts establish substantial Helmet/No Helmet imbalance and a smaller No
Vest/Safety Vest imbalance. They do **not** establish that imbalance caused
low recall. Compare validation FN rates, confidence-threshold curves, and
resolution results before deciding whether to retrain. If a training
experiment is approved later, test one controlled intervention at a time,
keep the validation/test split roles fixed, inspect false positives as well as
recall, and reserve the test set for final reporting.
