# Student Pass/Fail Prediction (UCI Student Performance)

Predicts whether a student passes the final grade (`G3`) from school, family, social and
earlier-grade attributes, using a Random Forest on the UCI
[Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance) data
(Portuguese secondary schools, Mathematics and Portuguese-language courses).

## Method

1. Download the UCI archive and concatenate the Portuguese (`student-por`) and Mathematics
   (`student-mat`) files.
2. Turn the final grade into a label: **pass = `G3 >= 8`**, otherwise fail. (The Portuguese
   grading scale is 0-20 with 10 as the usual pass mark, so 8 is a more lenient, project-specific
   cut-off.)
3. Fill missing values, label-encode binary columns, one-hot encode nominal columns, and
   standardize numeric features.
4. Random split 80 / 20 (seed 42), then train a Random Forest (200 trees).

## Original single-split result

On the 209 held-out students:

| | Predicted fail | Predicted pass |
| --- | ---: | ---: |
| **Actual fail** (21) | 14 | 7 |
| **Actual pass** (188) | 1 | 187 |

- Accuracy **96.2%**.
- Fail class: precision 14/15 = 0.93, recall 14/21 = 0.67.

## How to read this result

- **Class imbalance.** 90% of the test students pass, so always predicting "pass" already
  scores about 90%. The model's real value is in catching failures, and it finds two thirds of
  them.
- **Earlier grades are inputs.** The features include the period grades `G1` and `G2`, which
  are very strong predictors of `G3`. The model therefore mostly says "students who passed
  earlier terms pass again", not what social or demographic factors drive outcomes. A more
  useful early-warning model would drop `G1` and `G2`.
- **Possible overlap between files.** The UCI documentation notes that some students appear in
  both course files. After concatenating and randomly splitting, the same student can land in
  train and test, which can inflate the score.
- The original result uses one split and one seed, with no confidence interval.

## Stratified cross-validation

This evaluation uses only the UCI Portuguese-language course file (`student-por.csv`),
so students who appear in both course files cannot cross folds through the Mathematics
file. It uses five shuffled stratified folds (seed 42). The two Random Forests use the
notebook's 200-tree configuration, with preprocessing fit on each training fold. The
majority-class baseline also learns the most frequent class from each training fold.

The table pools held-out predictions across folds. Precision is reported as zero when
a model never predicts that class. These results describe one course in the original
two-school dataset. Earlier grades are available only after those grading periods.

| Pass mark | Model | Accuracy | Fail precision | Fail recall | Pass precision | Pass recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| G3 ≥ 8 | Majority class | 0.954 | 0.000 | 0.000 | 0.954 | 1.000 |
| G3 ≥ 8 | Random Forest | 0.963 | 1.000 | 0.200 | 0.963 | 1.000 |
| G3 ≥ 8 | Random Forest without G1/G2 | 0.955 | 1.000 | 0.033 | 0.955 | 1.000 |
| G3 ≥ 10 | Majority class | 0.846 | 0.000 | 0.000 | 0.846 | 1.000 |
| G3 ≥ 10 | Random Forest | 0.924 | 0.823 | 0.650 | 0.939 | 0.974 |
| G3 ≥ 10 | Random Forest without G1/G2 | 0.858 | 0.667 | 0.160 | 0.866 | 0.985 |

The [JSON report](eval/cross_validation.json) records class counts, confusion matrices,
per-fold metrics, dependency versions and the command used to run the evaluation.

```bash
python scripts/evaluate.py --output eval/cross_validation.json
```

## Run

```bash
git clone https://github.com/sachin2967/student-performance-prediction.git
cd student-performance-prediction
python -m venv .venv && source .venv/bin/activate   # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter notebook Student_Performance.ipynb   # downloads the data from UCI on first run
```

## License

MIT. The dataset is from the UCI Machine Learning Repository; see its page for terms and the
original citation (Cortez and Silva, 2008).
