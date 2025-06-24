#1.IMPORT LIBRARIES 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#2.LOAD OF RESULTS
print("\n=== 1. Load of Results ===")
df_results = pd.read_csv('/home/kevin/Desktop/Kevin/3.MachineLearning/1.FundamentalsML/2.HousePricePrediction/2.4EvaluationMetrics/df_results.csv')
print("Loads of Results:")
print(df_results)

#3.ANALYSIS OF OVERFITTING AND UNDERFITTING
def analyze_overfitting(row):
    train_r2 = row['Train R²']
    test_r2 = row['Test R²']
    train_rmse = row['Train RMSE']
    test_rmse = row['Test RMSE']
    
    #3.1 Simple rule: if there is a considerable difference between train and test
    if train_r2 > 0.90 and (train_r2 - test_r2) > 0.15:
        return "⚠️ Overfitting"
    elif train_r2 < 0.5 and test_r2 < 0.5:
        return "⚠️ Underfitting"
    else:
        return "✅ Good Balance"

df_results['Observation'] = df_results.apply(analyze_overfitting, axis=1)

#4.SELECTION OF THE BEST MODEL
print("\n=== 2. Analysis of Overfitting/Underfitting ===")
print(df_results)

#4.1 Sort by Test RMSE (better model has lower RMSE)
df_results_sorted = df_results.sort_values(by='Test RMSE')

#4.2 Select the best model (lowest Test RMSE, can be adjusted if you prefer to use R² as the primary criterion)
best_model = df_results_sorted.iloc[0]['Model']

print("\n Best Model Selected:", best_model)

#5.COMPARATIVE VISUALIZATION
print("\n=== 3. Graphical Comparison ===")

plt.figure(figsize=(14, 6))

x = np.arange(len(df_results['Model']))
width = 0.35  # Width of bars

#6.RMSE Graph (Train vs Test)
plt.subplot(1, 2, 1)
plt.bar(x - width/2, df_results['Train RMSE'], width=width, label='Train RMSE', color='skyblue')
plt.bar(x + width/2, df_results['Test RMSE'], width=width, label='Test RMSE', color='blue')
plt.xticks(x, df_results['Model'], rotation=45)
plt.ylabel('RMSE')
plt.title('Comparation of RMSE (Train vs Test)')
plt.legend()

#7.HIGHLIGHT THE BEST MODEL
best_index = df_results[df_results['Model'] == best_model].index[0]
plt.gca().get_xticklabels()[best_index].set_color('red')
plt.gca().get_xticklabels()[best_index].set_fontweight('bold')

#8.R² GRAPH(TRAIN vs TEST)
plt.subplot(1, 2, 2)
plt.bar(x - width/2, df_results['Train R²'], width=width, label='Train R²', color='lightgreen')
plt.bar(x + width/2, df_results['Test R²'], width=width, label='Test R²', color='green')
plt.xticks(x, df_results['Model'], rotation=45)
plt.ylabel('R²')
plt.title('Comparation of R² (Train vs Test)')
plt.legend()

#9.HIGHLIGHT THE BEST MODEL 
plt.gca().get_xticklabels()[best_index].set_color('red')
plt.gca().get_xticklabels()[best_index].set_fontweight('bold')

plt.tight_layout()
plt.show()

#10.FINAL SUMMARY
print("\n=== 4. Final Summary ===")
print(df_results)

print(f"\n Conclusion: The best model selected is: **{best_model}**")
print("This model has the best balance between accuracy and error on test data.")
