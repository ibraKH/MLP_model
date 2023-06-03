# importing all the libraries 

import pandas as pd
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.cluster import KMeans


# Read the input data
data = pd.read_csv("./data/data.csv")

# Preprocess the data
label_encoder = LabelEncoder()
data['sales'] = label_encoder.fit_transform(data['sales'])
data['salary'] = label_encoder.fit_transform(data['salary'])

# Show only 10 samples of the data
data.sample(10)

# Counts employees who left the company
data.left.value_counts()

# Distribution of features : 
features = ['number_project', 'time_spend_company', 'Work_accident', 'left', 'promotion_last_5years', 'sales', 'salary']

fig, axs = plt.subplots(4, 2, figsize=(10, 15))

for i, j in enumerate(features):
    ax = axs[i // 2, i % 2]
    sns.countplot(x=j, data=data, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_title("No. of Employees")

# Remove overlapping axes
plt.tight_layout()
plt.show()

colors = ['#00C4FF', '#FF6D60']

fig, axs = plt.subplots(4, 2, figsize=(10, 15))

for i, j in enumerate(features):
    ax = axs[i // 2, i % 2]
    sns.countplot(x=j, data=data, hue='left', palette=colors, hue_order=[0, 1], ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_title("No. of Employees")

# Remove overlapping axes
plt.tight_layout()
plt.show()

# K-means algorithm : 
left_emp = data[data['left'] == 1][['satisfaction_level', 'last_evaluation']]
kmeans = KMeans(n_clusters=3, random_state=0, n_init=10).fit(left_emp)
left_emp['label'] = kmeans.labels_
plt.scatter(left_emp['satisfaction_level'], left_emp['last_evaluation'], c=left_emp['label'], cmap='Set3')
plt.xlabel('Satisfaction Level')
plt.ylabel('Last Evaluation')
plt.title('3 Clusters of Employees Who Left')
plt.show()

# Spliting data into Feature and target
X=data[['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company',
        'Work_accident', 'promotion_last_5years', 'sales', 'salary']]
y=data['left']


# Import train_test_split function 
from sklearn.model_selection import train_test_split

# Split dataset into 70% training set and 30% test set.  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Import MLPClassifer 
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(hidden_layer_sizes=(6,5),
                    random_state=5,
                    verbose=True,
                    learning_rate_init=0.01)

# Fit data onto the model
clf.fit(X_train,y_train)

# Make prediction on test dataset
ypred=clf.predict(X_test)

# Import scikit-learn metrics module for calculation
from sklearn import metrics

# Model Accuracy
print("Accuracy:",metrics.accuracy_score(y_test, ypred))

# Model Precision
print("Precision:",metrics.precision_score(y_test, ypred))

# Model Recall
print("Recall:",metrics.recall_score(y_test, ypred))

# Model F-score
print("F-score:",metrics.f1_score(y_test, ypred))
