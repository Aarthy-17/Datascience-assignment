# Datascience-assignment for iris flower classification
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the iris dataset
iris = load_iris()
X = iris.data  
y = iris.target  

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Decision Tree Classifier
clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
clf.fit(X_train, y_train)

# Load new examples from CSV file
new_examples_df = pd.read_csv("C:\\Users\\aarth\\Downloads\\IRIS (1).csv")

# Predict for each new example
for index, row in new_examples_df.iterrows():
    new_example = [row["sepal_length"], row["sepal_width"], row["petal_length"], row["petal_width"]]
    prediction = clf.predict([new_example])
    print(f"Prediction for new example {new_example}: {iris.target_names[prediction][0]}")

# Accuracy on the test set
accuracy = clf.score(X_test, y_test)
print(f"Accuracy on the test set: {accuracy:.2f}")

# Plot the decision tree
plt.figure(figsize=(15, 10))
plot_tree(clf, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title("Decision Tree Based on ID3 Algorithm")
plt.show()



# Export and print the decision tree rules
tree_rules = export_text(clf, feature_names=iris.feature_names)
print("Decision Tree Rules:\n", tree_rules)
#**Run it in any jupyter notebook**


