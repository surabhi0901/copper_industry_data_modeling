import pickle
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, roc_auc_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import streamlit as st

#Uploading the data



#Define target and features for regression
X_reg = st.session_state.data.drop(columns=['selling_price'])
y_reg = st.session_state.data['selling_price']

#Split the data for regression
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

#Define regression models
regression_models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree Regressor": DecisionTreeRegressor(),
    "Random Forest Regressor": RandomForestRegressor(),
    "Gradient Boosting Regressor": GradientBoostingRegressor(),
    "XGBoost Regressor": XGBRegressor(),
    "KNeighbors Regressor": KNeighborsRegressor()
}

# Train, predict, and evaluate each regression model, and save them
for name, model in regression_models.items():
    model.fit(X_train_reg, y_train_reg)
    y_pred_reg = model.predict(X_test_reg)
    mse = mean_squared_error(y_test_reg, y_pred_reg)
    rmse = mse ** 0.5
    r2 = r2_score(y_test_reg, y_pred_reg)
    print(f"Model: {name}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"R2 Score: {r2}")
    print("----------")
    with open(f'/mnt/data/{name.replace(" ", "_").lower()}_regressor.pkl', 'wb') as f:
        pickle.dump(model, f)

# Define target and features for classification
classification_data = st.session_state.data[st.session_state.data['status'].isin(['WON', 'LOST'])]
X_cls = classification_data.drop(columns=['status'])
y_cls = classification_data['status'].map({'WON': 1, 'LOST': 0})

# Split the data for classification
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)

# Define classification models
classification_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, solver='liblinear'),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB()
}

# Train, predict, and evaluate each classification model, and save them
for name, classifier in classification_models.items():
    classifier.fit(X_train_cls, y_train_cls)
    y_pred_cls = classifier.predict(X_test_cls)
    accuracy = accuracy_score(y_test_cls, y_pred_cls)
    report = classification_report(y_test_cls, y_pred_cls, output_dict=True)
    auc = roc_auc_score(y_test_cls, classifier.predict_proba(X_test_cls)[:, 1])
    print(f"Classifier: {name}")
    print(f"Accuracy: {accuracy}")
    print(f"AUC: {auc}")
    print("Classification Report:")
    print(classification_report(y_test_cls, y_pred_cls))
    print("----------")
    with open(f'/mnt/data/{name.replace(" ", "_").lower()}_classifier.pkl', 'wb') as f:
        pickle.dump(classifier, f)

# Save the scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_reg)
X_test_scaled = scaler.transform(X_test_reg)
with open('/mnt/data/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)