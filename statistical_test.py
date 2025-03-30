from scipy.stats import ttest_ind
from sklearn.metrics import accuracy_score

def get_statistical_significance(y_test, y_pred_initial, y_pred_optimized):
    # Compute accuracy scores
    acc_initial = accuracy_score(y_test, y_pred_initial)
    acc_optimized = accuracy_score(y_test, y_pred_optimized)

    # Perform statistical test (paired t-test)
    t_stat, p_value = ttest_ind(y_pred_initial, y_pred_optimized)

    print(f"Initial Model Accuracy: {acc_initial:.4f}")
    print(f"Optimized Model Accuracy: {acc_optimized:.4f}")
    print(f"T-test Statistic: {t_stat:.4f}, P-value: {p_value:.4f}")

    if p_value < 0.05:
        print("The improvement in model accuracy is statistically significant.")
    else:
        print("No significant improvement in model accuracy.")