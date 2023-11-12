# Required installations (use pip or conda):
# pip install streamlit scikit-learn matplotlib

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.title("Random Forest Interactive Demonstrator")

# Create synthetic dataset with a reasonable number of features
n_features = 10
X, y = make_classification(n_samples=500, n_features=n_features, n_informative=5, n_redundant=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

st.sidebar.header("Parameters")
n_trees = st.sidebar.slider("Number of trees", 1, 100, 10)
max_features = st.sidebar.slider("Max features for split", 1, n_features, int(n_features / 2))
button = st.sidebar.button("Generate Forest")

# Plot showing sample trees
def plot_sample_trees(X, y, num_trees, max_features):
    fig, axes = plt.subplots(1, num_trees, figsize=(20, 4))
    if num_trees == 1:
        axes = [axes]
    for i in range(num_trees):
        indices = np.random.choice(range(X.shape[0]), size=int(X.shape[0] * 0.8), replace=True)
        sample_X, sample_y = X[indices], y[indices]
        clf = RandomForestClassifier(n_estimators=1, max_features=max_features, bootstrap=True)
        clf.fit(sample_X, sample_y)
        axes[i].scatter(sample_X[:, 0], sample_X[:, 1], c=sample_y, cmap='coolwarm', s=15)
        axes[i].set_title(f'Tree {i+1}')
        axes[i].set_xlabel('Feature 1')
        axes[i].set_ylabel('Feature 2')
    plt.tight_layout()
    st.pyplot(fig)


def plot_bias_variance(n_trees, max_features, X_train, y_train, X_test, y_test):
    bias = []
    variance = []
    trees_range = range(1, n_trees + 1)
    for n in trees_range:
        clf = RandomForestClassifier(n_estimators=n, max_features=max_features, random_state=42)
        clf.fit(X_train, y_train)
        train_pred = clf.predict(X_train)
        test_pred = clf.predict(X_test)
        bias.append(1 - accuracy_score(y_train, train_pred))
        variance.append(1 - accuracy_score(y_test, test_pred))
    
    # Plot for Bias
    fig_bias, ax_bias = plt.subplots()
    ax_bias.plot(trees_range, bias, '-r', label='Bias (1 - Accuracy)')
    ax_bias.set_xlabel('Number of Trees')
    ax_bias.set_ylabel('Bias (1 - Accuracy)')
    ax_bias.set_title('Effect of Number of Trees on Bias')
    ax_bias.legend()
    st.pyplot(fig_bias)

    # Plot for Variance
    fig_variance, ax_variance = plt.subplots()
    ax_variance.plot(trees_range, variance, '-b', label='Variance (1 - Accuracy)')
    ax_variance.set_xlabel('Number of Trees')
    ax_variance.set_ylabel('Variance (1 - Accuracy)')
    ax_variance.set_title('Effect of Number of Trees on Variance')
    ax_variance.legend()
    st.pyplot(fig_variance)


# Show new predictions
def show_predictions(n_trees, max_features, X, y):
    clf = RandomForestClassifier(n_estimators=n_trees, max_features=max_features, random_state=42)
    clf.fit(X, y)
    st.subheader("New Predictions")
    random_sample = X[np.random.choice(X.shape[0], 1, replace=False), :]
    prediction = clf.predict(random_sample)
    st.write(f"Random sample features: {random_sample}")
    st.write(f"The predicted class is: {prediction[0]}")

# On button click, generate plots and show predictions
if button:
    plot_sample_trees(X_train, y_train, 3, max_features)  # Plot sample trees from the forest
    plot_bias_variance(n_trees, max_features, X_train, y_train, X_test, y_test)  # Plot bias and variance
    show_predictions(n_trees, max_features, X_train, y_train)  # Show new predictions
