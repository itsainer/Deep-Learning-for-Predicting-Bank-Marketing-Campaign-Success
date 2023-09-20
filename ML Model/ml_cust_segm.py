"""
Continuing from deep learning model (dl_cust_segm.py)...

Task: Develop a machine learning model to predict the outcome of the marketing campaign (term_deposit_subscribed). 
Model trained using: RandomForestClassifier and SVC
"""

#%%
# Import packages
import pandas as pd

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
# %%
# Load the cleaned dataset from previous dl model
df = pd.read_csv('df_customer.csv')
# %%
# Define features and target
x = df.drop(['term_deposit_subscribed'], axis=1).copy()
y = df['term_deposit_subscribed']

# Split dataset into training and testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=41, stratify=y)
# %%
# Define model names and corresponding pipeline steps in a dictionary
pipelines = {
    "SS + RF": Pipeline([('SS', StandardScaler()), ('RF', RandomForestClassifier())]),
    "MM + RF": Pipeline([('MM', MinMaxScaler()), ('RF', RandomForestClassifier())]),
    "SS + SVC": Pipeline([('SS', StandardScaler()), ('SVC', SVC())]),
    "MM + SVC": Pipeline([('MM', MinMaxScaler()), ('SVC', SVC())])
}
# %%
# Train all the pipelines
results_dict = {}
best_score = 0.0
best_pipe = ''

for name, pipe in pipelines.items():
    print(f'Training {name} ..')
    pipe.fit(x_train, y_train)
    
    y_pred = pipe.predict(x_test)

    results_dict[name] = [accuracy_score(y_test, y_pred),
                          precision_score(y_test, y_pred),
                          recall_score(y_test, y_pred),
                          f1_score(y_test, y_pred, average='weighted'),
                          roc_auc_score(y_test, y_pred)]

    if pipe.score(x_test, y_test) > best_score:
        best_score = pipe.score(x_test, y_test)
        best_pipe = name

# Display the results of training
print(f'The best model is {best_pipe} with accuracy score of {best_score}')
print("\nClassification report: \n", classification_report(y_test, y_pred))
print("\nConfusion matrix: \n", confusion_matrix(y_test, y_pred))
# %%
# Convert results dictionaries into a dataframe
results_df = pd.DataFrame(results_dict, index=['Accuracy','Precision','Recall','F1 Score','ROC_AUC']).T
# %%
# Plot evaluation based on models
results_df.sort_values(by='Accuracy').style.background_gradient(cmap='Blues')
# %%
