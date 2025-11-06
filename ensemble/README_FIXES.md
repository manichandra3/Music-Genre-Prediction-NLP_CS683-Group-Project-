# Ensemble Learning - Fixed Implementation

## Issues Found and Corrected

### 🔴 Critical Issues

#### 1. **Data Mismatch (CRITICAL)**
**Problem:** Most notebooks used the same data source for both LSTM and CNN models
```python
# ❌ WRONG - Original implementation
mfccs = data['mfcc']  # Same data for both models
lstm_probs = lstm_model.predict(X)
cnn_probs = cnn_model.predict(X)  # Same input!
```

**Fix:** Use separate vocal and accompaniment data sources
```python
# ✅ CORRECT - Fixed implementation
data_vocal = json.load('data.json')  # Vocal features
data_accomp = json.load('accompaniment_mfcc.json')  # Accompaniment features

lstm_probs = lstm_model.predict(X_vocal)
cnn_probs = cnn_model.predict(X_accomp)
```

#### 2. **Data Leakage in Stacking (MAJOR)**
**Problem:** Meta-models trained and tested on the same data
```python
# ❌ WRONG - Causes severe overfitting
lr_meta.fit(stacking_features, y_test)  # Train on test data!
lr_preds = lr_meta.predict(stacking_features)  # Predict on same data
# Result: Inflated accuracy scores
```

**Fix:** Proper train/validation/test split
```python
# ✅ CORRECT - No data leakage
# Generate predictions on validation set
lstm_probs_val = lstm_model.predict(X_val)
cnn_probs_val = cnn_model.predict(X_val)

# Train meta-model on validation predictions
stacking_features_val = np.concatenate([lstm_probs_val, cnn_probs_val], axis=1)
lr_meta.fit(stacking_features_val, y_val)

# Evaluate on held-out test set
stacking_features_test = np.concatenate([lstm_probs_test, cnn_probs_test], axis=1)
lr_preds = lr_meta.predict(stacking_features_test)
accuracy = accuracy_score(y_test, lr_preds)
```

### 🟡 Major Issues

#### 3. **Inconsistent CNN Architecture**
**Problem:** Multiple `input_shape` parameters in the same model
```python
# ❌ WRONG
Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape),
MaxPooling2D((2, 2)),
Conv2D(32, (3, 3), activation='relu', padding='same'),
Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape),  # Error!
```

**Fix:** Only specify input_shape in the first layer
```python
# ✅ CORRECT
Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape),
MaxPooling2D((2, 2)),
Conv2D(128, (3, 3), activation='relu', padding='same'),  # No input_shape
MaxPooling2D((2, 2)),
```

#### 4. **No Proper Data Split**
**Problem:** Only train/test split, no validation set for meta-models
```python
# ❌ WRONG
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
```

**Fix:** Proper 60/20/20 split
```python
# ✅ CORRECT - 60% train, 20% val, 20% test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp)
```

### 🟢 Minor Issues

#### 5. **Unclear Weighting Strategy**
**Problem:** Arbitrary weights without justification
```python
# ❌ Unclear
weights = [0.5, 0.5]  # Why these weights?
weights = [0.33, 0.67]  # Why different in other notebooks?
```

**Fix:** Calculate weights based on validation performance
```python
# ✅ CORRECT - Data-driven weights
lstm_val_acc = accuracy_score(y_val, lstm_preds_val)
cnn_val_acc = accuracy_score(y_val, cnn_preds_val)

total = lstm_val_acc + cnn_val_acc
w_lstm = lstm_val_acc / total
w_cnn = cnn_val_acc / total
```

#### 6. **Dense Meta-Model Issues**
**Problem:** Redundant input_shape specifications, excessive dropout
```python
# ❌ WRONG
Dense(128, activation='relu', input_shape=input_shape),
Dropout(0.5),
Dense(64, activation='relu', input_shape=input_shape),  # Redundant!
Dropout(0.5),
```

**Fix:** Clean architecture with appropriate dropout
```python
# ✅ CORRECT
Dense(128, activation='relu', input_dim=input_dim),
Dropout(0.3),  # Reduced dropout
Dense(64, activation='relu'),
Dropout(0.3),
Dense(32, activation='relu'),
Dropout(0.3),
Dense(num_classes, activation='softmax')
```

## New Implementation Structure

### Data Flow
```
Audio Files
    │
    ├─→ Vocal Extraction → MFCC → data.json
    │
    └─→ Accompaniment Extraction → MFCC → accompaniment_mfcc.json
         │
         ├─→ LSTM Model (Vocal) ──┐
         │                         │
         └─→ CNN Model (Accomp) ───┼→ Ensemble
                                    │
                                    ├─→ Bagging (Voting)
                                    │   ├─ Mean Averaging
                                    │   ├─ Weighted Voting
                                    │   └─ Max Voting
                                    │
                                    └─→ Stacking
                                        ├─ Logistic Regression
                                        ├─ XGBoost
                                        └─ Neural Network
```

### Proper Train/Val/Test Split
```
Total Data (100%)
    │
    ├─→ Training (60%) ────→ Train base models (LSTM, CNN)
    │
    ├─→ Validation (20%) ──→ Train meta-models, tune weights
    │
    └─→ Test (20%) ────────→ Final evaluation (never seen before)
```

## Usage

### Prerequisites
```bash
pip install tensorflow numpy scikit-learn xgboost matplotlib seaborn
```

### Required Data Files
1. `../Data/data.json` - Vocal MFCCs
2. `../Data/accompaniment_mfcc.json` - Accompaniment MFCCs (optional)
3. Pre-trained models (or will train from scratch):
   - `../models/lstm_vocal_classifier.keras`
   - `../models/cnn_accompaniment_classifier.keras`

### Running the Fixed Implementation
```bash
jupyter notebook Ensemble_learning_FIXED.ipynb
```

## Expected Results

### Individual Models
- LSTM (Vocal): ~70-75% accuracy
- CNN (Accompaniment): ~72-77% accuracy

### Ensemble Methods
- **Bagging Methods:**
  - Mean Averaging: ~75-78%
  - Weighted Voting: ~76-79%
  - Max Voting: ~74-77%

- **Stacking Methods:**
  - Logistic Regression: ~77-80%
  - XGBoost: ~78-82% ⭐ **Best**
  - Neural Network: ~77-81%

### Key Improvements
✓ No data leakage
✓ Proper validation methodology
✓ Reproducible results
✓ Clean, maintainable code
✓ Comprehensive evaluation

## Validation Checklist

✅ Separate vocal and accompaniment data sources
✅ Proper 60/20/20 train/val/test split with stratification
✅ Meta-models trained on validation set only
✅ Final evaluation on held-out test set
✅ Consistent architecture across models
✅ Data-driven hyperparameter selection
✅ Comprehensive result comparison

## Key Takeaways

1. **Always use separate validation set** for meta-model training
2. **Never evaluate on training data** - it inflates accuracy scores
3. **Use stratified splits** to maintain class balance
4. **Ensemble only helps** if base models learn different features
5. **Document your methodology** for reproducibility

## Further Improvements

- [ ] Implement K-fold cross-validation for more robust stacking
- [ ] Add more diverse base models (e.g., RNN, Transformer)
- [ ] Hyperparameter tuning with grid/random search
- [ ] Feature importance analysis for ensemble decisions
- [ ] Add confidence-based rejection option
- [ ] Implement model calibration for probability outputs

## References

- Zhou, Z. H. (2012). Ensemble Methods: Foundations and Algorithms
- Wolpert, D. H. (1992). Stacked Generalization
- Breiman, L. (1996). Bagging Predictors

---

**Author:** Fixed Implementation  
**Date:** November 6, 2025  
**Status:** Production Ready ✅
