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

## Result

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
- Single split, single seed, no cross-validation or confidence interval.

## Run

```bash
git clone https://github.com/sachin2967/student-performance-prediction.git
cd student-performance-prediction
python -m venv .venv && source .venv/bin/activate   # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter notebook Student_Performance.ipynb   # downloads the data from UCI on first run
```

## Next steps

1. Remove `G1`/`G2` and compare; report precision/recall per class.
2. Split by student so no one appears in both train and test; add cross-validation.
3. Use the standard pass mark (`G3 >= 10`) and compare against a majority-class baseline.

## License

MIT. The dataset is from the UCI Machine Learning Repository; see its page for terms and the
original citation (Cortez and Silva, 2008).
