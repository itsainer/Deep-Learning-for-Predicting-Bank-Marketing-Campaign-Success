"""
The purpose of marketing campaign is to collect customer’s needs and overall satisfaction. There are a few essential aspects of the 
marketing campaign namely, customer segmentation, promotional strategy, and etc. Correctly identified strategy may help to expand and 
grow the bank’s revenue. 

Task: Develop a deep learning model to predict the outcome of the marketing campaign (term_deposit_subscribed). 
"""

#%%
# Import packages
import os
import pickle
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
#%%
# Load the dataset
df = pd.read_csv('train.csv')
#%%
# Data Cleaning
df.info()
# 1. Drop column id, days_since_prev_campaign_contact and num_contacts_prev_campaign
df.drop(df.loc[:, ['id','days_since_prev_campaign_contact','num_contacts_prev_campaign']].columns, axis=1, inplace=True)

# 2. Columns with missing values:
# (A) Fill marital, personal loan with unknown
df["marital"] = df["marital"].fillna("unknown")
df["personal_loan"] = df["personal_loan"].fillna("unknown")

# (B) Check outliers for other 4 columns with nan:
col_check = ['customer_age', 'balance', 'last_contact_duration', 'num_contacts_in_campaign']
fig_num = 1        
for col in df.select_dtypes(include=[np.number]).columns:
    if col in col_check:
        df.boxplot(figsize=(10,5))
        fig_num = fig_num + 1        

# (C) Clip outliers in balance column within a range
df['balance'] = df['balance'].clip(lower=-8020, upper=40000)

# 3. Fill balance with min value
df["balance"] = df["balance"].fillna(-8020)

# 4. Fill other 3 columns with median
# (A) Customer age
married  = df[df["marital"] == "married"]["customer_age"].median()
single   = df[df["marital"] == "single"]["customer_age"].median()
divorced = df[df["marital"] == "divorced"]["customer_age"].median()
unknown  = df[df["marital"] == "unknown"]["customer_age"].median()

for i in range(len(df)):
    if np.isnan(df["customer_age"][i]):
        if df["marital"][i] == "married":
            df["customer_age"][i] = round(married)
        if df["marital"][i] == "single":
            df["customer_age"][i] = round(single)
        if df["marital"][i] == "divorced":
            df["customer_age"][i] = round(divorced)
        if df["marital"][i] == "unknown":
            df["customer_age"][i] = round(unknown)

# (B) Last contact duration, (C) Num contacts in campaign
for cols in ['last_contact_duration', 'num_contacts_in_campaign']:
  df[cols] = df[cols].fillna(df[cols].median())
  df[cols] = df[cols].fillna(df[cols].median())
#%%
# Data Preprocessing

# Separate continuous and categorical data
cont_df = df.select_dtypes(include=['float64','int32','int64'])
cat_df = df.select_dtypes(include='object')
#%%
# Encode categorical features using Label Encoding
col_to_encode = cat_df
label_encoder = LabelEncoder()
for column in col_to_encode:
    df[column] = label_encoder.fit_transform(df[column])
#%%
# Perform Train-Test-Split

# Split the dataset into features (X) and the target variable (y)
x = df.drop(['term_deposit_subscribed'], axis=1).copy()
y = df['term_deposit_subscribed']
#%%
# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=13, stratify=y)
#%%
# Display shape of training and testing sets
print("x_train: ", x_train.shape)
print("x_test: ", x_test.shape)
print("y_train: ", y_train.shape)
print("y_test: ", y_test.shape)
#%%
# Data Scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
#%%
# Model Development

# To visualize training results by Tensorboard
base_log_path = r"tensorboard_log"
log_path = os.path.join(base_log_path, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb = tf.keras.callbacks.TensorBoard(log_path)
#%%
# Build the dl model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(x_train.shape[1],)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')  #Binary classification output
])
#%%
# Define epochs and callback
MAX_EPOCHS = 20
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=2,
                                                    mode='min')
#%%
# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#%%
# Train the model
history = model.fit(x_train, y_train, epochs=MAX_EPOCHS, batch_size=32, validation_split=0.2, callbacks=[tb, early_stopping])
#%%
# Evaluate the model
y_pred = model.predict(x_test)
y_pred = (y_pred > 0.5)                  #Convert probabilities to binary predictions
#%%
# Display evaluations
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("Precision score: ", precision_score(y_test, y_pred))
print("Recall score: ", recall_score(y_test, y_pred))
print("F1 score: ", f1_score(y_test, y_pred, average='weighted'))
print("ROC AUC score: ", roc_auc_score(y_test, y_pred))
print("\nClassification report: \n", classification_report(y_test, y_pred))
print("Confusion matrix: \n", confusion_matrix(y_test, y_pred))
#%%
# Plot confusion matrix using seaborn
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, cmap='Greens')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()
#%%
# Display model summary
model.summary()
# Display model structure
tf.keras.utils.plot_model(model)
#%%
# Plot model performance
fig = plt.figure()
plt.plot(history.history['accuracy'], color='teal', label='accuracy')
plt.plot(history.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc='upper right')
plt.show()

fig = plt.figure()
plt.plot(history.history['loss'], color='teal', label='loss')
plt.plot(history.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc='upper right')
plt.show()
# %%
# Save model
model.save(os.path.join('DL Model', 'dl_model.h5'))
#%%
# Save scaler in .pkl
# Specify the file path
scaler_filename = 'scaler.pkl'

with open(scaler_filename, 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)
# %%
# Export df
# Define the file path
file_path = 'df_customer.csv' 

# Export the cleaned DataFrame to a CSV file
df.to_csv(file_path, index=False)

print(f'dataframe (df) has been saved to {file_path}')
# %%
# To load scaler.pkl
# Open the .pkl file in binary read mode and deserialize the scaler
with open(scaler_filename, 'rb') as scaler_file:
    loaded_scaler = pickle.load(scaler_file)