# Random Forest Report – Heart Disease

## Hyperparameters

- Untuned: default settings
- Tuned RF via GridSearchCv

```bash
{
    'n_estimators': 200,
    'max_depth': 5,
    'min_samples_split': 5,
    'max_features': 'log2'
}
```

## Evaluation Results

- RF Accuracy: 0.8206521739130435
- RF AUC: 0.9106886657101865

Confusion Matrix:
![Confusion Matrix](/Users/manuelrodriguezsutil/Developer/health-risk/visuals/heart_disease/confusion_matrix_rf.png)

- Tuned RF Accuracy: 0.8206521739130435
- Tuned RF AUC: 0.9177427068388331

Confusion Matrix:
![Confusion Matrix](/Users/manuelrodriguezsutil/Developer/health-risk/visuals/heart_disease/tuned_confusion_matrix_rf.png)

## Feature Importance

![Feature Importance](/Users/manuelrodriguezsutil/Developer/health-risk/visuals/heart_disease/rf_importance.png)

## Interpretation

- The tuned Random Forest slightly outperforms logistic regression on AUC, while maintaining nearly identical accuracy.
- It also reduces false negatives (from 10 → 9), a key metric in clinical risk modeling.

## Conclusion
- The Random Forest outperforms Logistic Regression on AUC.
- Tuning did improve performace slightly on this dataset.
