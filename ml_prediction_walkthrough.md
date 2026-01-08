# Machine Learning Prediction Walkthrough: Predicting Next Competition Lifts

## Overview
This guide walks through the **process** of building a regression model to predict a weightlifter's next competition best squat, bench, and deadlift. This is a high-level roadmap of the steps involved, not implementation details.

---

## Step 1: Understand Your Data Structure

### What You Have
- **Entries**: Each row represents one lifter's performance at one competition
  - Key columns: `LifterID`, `MeetID`, `Best3SquatKg`, `Best3BenchKg`, `Best3DeadliftKg`
  - Other useful columns: `Age`, `BodyweightKg`, `WeightClassKg`, `Sex`, `Equipment`, `Date`
  
- **Meets**: Each row represents one competition
  - Key columns: `MeetID`, `Date` (links to entries via `MeetID`)

### The Goal
For each lifter, predict their **next competition's** best squat, bench, and deadlift based on their **previous competition history**.

---

## Step 2: Prepare the Data for Time-Series Analysis

### 2.1 Combine Entries with Meet Dates
- Merge entries with meet data using `MeetID` to get the competition date for each entry
- This gives you a timeline: each lifter's competitions ordered by date

### 2.2 Group by Lifter
- Group all entries by `LifterID`
- Sort each lifter's entries by date (chronologically)
- This creates a "competition history" for each lifter

### 2.3 Filter Valid Lifters
- Keep only lifters who have **at least 2 competitions** (you need history to predict the future)
- Optionally filter by other criteria (e.g., only SBD events, only Raw equipment, etc.)

---

## Step 3: Create Training Examples (The "Sliding Window" Approach)

### The Core Idea
For each lifter with N competitions, you can create N-1 training examples:
- **Example 1**: Use competition 1 to predict competition 2
- **Example 2**: Use competitions 1-2 to predict competition 3
- **Example 3**: Use competitions 1-3 to predict competition 4
- And so on...

### What Goes Into Each Training Example

**Input Features (X)** - What the model sees:
- Historical lift values: Previous best squat, bench, deadlift
- Time-based features: Number of competitions, time since last competition, age progression
- Context features: Weight class, bodyweight, equipment type, sex
- Trend features: Rate of improvement, consistency, personal records

**Target Variables (y)** - What you're trying to predict:
- Next competition's `Best3SquatKg`
- Next competition's `Best3BenchKg`
- Next competition's `Best3DeadliftKg`

---

## Step 4: Feature Engineering

### 4.1 Historical Performance Features
- Previous best squat/bench/deadlift (from last competition)
- All-time personal records for each lift
- Average of last N competitions
- Standard deviation (consistency measure)

### 4.2 Time-Based Features
- Days/months since last competition
- Number of total competitions
- Age at each competition
- Time between competitions

### 4.3 Progression Features
- Rate of improvement (change per competition)
- Whether lifter is improving, declining, or stable
- Number of competitions since last personal record

### 4.4 Context Features
- Weight class category
- Bodyweight at competition
- Equipment type (Raw, Wraps, etc.)
- Sex (M/F)
- Division (Open, Junior, etc.)

---

## Step 5: Handle Missing Data and Edge Cases

### 5.1 Missing Values
- Some entries may have missing lift values (e.g., DQ, no-show)
- Decide: drop these entries, or impute values (fill with reasonable defaults)

### 5.2 Inconsistent Data
- Some lifters may have gaps in their competition history
- Some may compete in different weight classes over time
- Decide how to handle these cases

### 5.3 Data Quality
- Remove or flag unrealistic values (e.g., negative lifts, impossibly high values)
- Handle outliers appropriately

---

## Step 6: Split Your Data

### 6.1 Train/Validation/Test Split
- **Training set**: Used to teach the model (e.g., 70%)
- **Validation set**: Used to tune hyperparameters and check progress (e.g., 15%)
- **Test set**: Used for final evaluation only, untouched until the end (e.g., 15%)

### 6.2 Important: Time-Based Splitting
- **Don't randomly split** - split by date!
- Use older competitions for training, newer ones for testing
- This simulates real-world prediction (predicting future based on past)

### 6.3 Alternative: Leave-One-Lifter-Out
- For each lifter, use all their competitions except the last one for training
- Use their last competition for testing
- This ensures the model hasn't "seen" the lifter's final performance during training

---

## Step 7: Choose Your Model Architecture

### 7.1 Single Model vs. Multiple Models
- **Option A**: One model that predicts all three lifts simultaneously (multitask learning)
- **Option B**: Three separate models (one for squat, one for bench, one for deadlift)
- **Option C**: One model per lift type (simpler, easier to interpret)

### 7.2 Model Types to Consider
- **Linear Regression**: Simple baseline, interpretable
- **Random Forest**: Handles non-linear relationships, feature importance
- **Gradient Boosting** (XGBoost, LightGBM): Often best performance
- **Neural Networks**: Can capture complex patterns, but needs more data
- **Time-Series Models**: LSTM/GRU if treating as sequence data

---

## Step 8: Train the Model

### 8.1 Loss Function
- For regression: Mean Squared Error (MSE) or Mean Absolute Error (MAE)
- Consider weighted loss if you care more about certain lifts

### 8.2 Training Process
- Feed training examples (features → target lifts) to the model
- Model learns patterns: "If a lifter squatted 200kg last time and is 25 years old, they'll likely squat 205kg next time"
- Adjust model parameters to minimize prediction error

### 8.3 Hyperparameter Tuning
- Try different model settings (learning rate, tree depth, etc.)
- Use validation set to evaluate which settings work best
- Don't use test set for this - save it for final evaluation!

---

## Step 9: Evaluate Model Performance

### 9.1 Metrics to Calculate
- **Mean Absolute Error (MAE)**: Average error in kg (e.g., "off by 5kg on average")
- **Root Mean Squared Error (RMSE)**: Penalizes larger errors more
- **R² Score**: How much better than just predicting the average
- **Per-lift metrics**: Separate scores for squat, bench, deadlift

### 9.2 Visualizations
- Scatter plots: Predicted vs. Actual lifts
- Residual plots: Where the model makes errors
- Error distribution: Are errors normally distributed?

### 9.3 Real-World Interpretation
- If MAE is 10kg, that means on average you're off by 10kg
- Is this acceptable for your use case?
- Check if errors are consistent across different lifters, ages, weight classes

---

## Step 10: Make Predictions

### 10.1 For a New Lifter
- Gather their competition history
- Extract the same features you used in training
- Feed to model → get predictions for next competition

### 10.2 Uncertainty
- Consider providing prediction intervals (e.g., "likely between 200-210kg")
- Some models can estimate uncertainty, others cannot

---

## Step 11: Iterate and Improve

### 11.1 Analyze Errors
- Where does the model fail? (Young lifters? Experienced lifters? Specific weight classes?)
- What patterns did it miss?

### 11.2 Feature Engineering Round 2
- Add new features based on error analysis
- Remove features that don't help
- Try feature interactions (e.g., age × experience)

### 11.3 Model Improvements
- Try different model architectures
- Ensemble multiple models
- Add regularization to prevent overfitting

---

## Common Pitfalls to Avoid

1. **Data Leakage**: Don't use future information to predict the past
2. **Overfitting**: Model memorizes training data but fails on new lifters
3. **Ignoring Time**: Randomly splitting data instead of time-based splitting
4. **Missing Data**: Not handling missing values properly
5. **Scale Issues**: Not normalizing features (age in years vs. weight in kg)
6. **Too Few Examples**: Need enough lifters with enough competitions

---

## Summary: The Big Picture

```
Raw Data (Entries + Meets)
    ↓
Combine & Sort by Date
    ↓
Group by Lifter → Competition Histories
    ↓
Create Training Examples (sliding window)
    ↓
Engineer Features (historical, time-based, context)
    ↓
Split Data (train/validation/test)
    ↓
Train Model
    ↓
Evaluate Performance
    ↓
Make Predictions
    ↓
Iterate & Improve
```

---

## Next Steps

Once you understand this process, you can:
1. Start implementing data loading and preprocessing
2. Experiment with different feature engineering approaches
3. Try different model types
4. Evaluate what works best for your specific use case

Remember: The process is iterative. Start simple, get something working, then improve it step by step.
