# Random Forest Report – Diabetes

## Hyperparameters

- Untuned: default settings
- Tuned RF via GridSearchCv

```bash
{
    'n_estimators': 100,
    'max_depth': 5,
    'min_samples_split': 2,
    'max_features': 'sqrt'
}
```

## Evaluation Results

- RF Accuracy: 0.7727272727272727
- RF AUC: 0.8189814814814814

Confusion Matrix:
![Confusion Matrix](/Users/manuelrodriguezsutil/Developer/health-risk/visuals/diabetes/confusion_matrix_rf.png)

- Tuned RF Accuracy: 0.7272727272727273
- Tuned RF AUC: 0.8062962962962963

Tuned Confusion Matrix:
![Confusion Matrix](/Users/manuelrodriguezsutil/Developer/health-risk/visuals/diabetes/tuned_confusion_matrix_rf.png)

## Feature Importance

![Feature Importance](/Users/manuelrodriguezsutil/Developer/health-risk/visuals/diabetes/rf_importance.png)

## Interpretation

- The model’s most important predictors are clinically consistent with diabetes diagnosis.
- The high AUC (0.82) and decent accuracy (0.77) suggest reliable discrimination.
- False negatives are still a concern in medical applications, where missing a diabetic patient has consequences.
- The lack of performance improvement with tuning suggests the model is already well-calibrated given the feature space.

## Conclusion

- The Random Forest outperforms Logistic Regression on all predictive measures.
- Tuning did not improve performace, on the contrary, it reduced both accuracy and AUC slightly.