import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sb

pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 100)
pd.set_option('display.min_rows', 20)
pd.set_option('mode.chained_assignment', None)

df = pd.read_csv('zadanie2_dataset.csv').drop(columns=['ID', 'Model', 'Color', 'Left wheel'])
df = df.drop_duplicates()

df['Levy'] = df['Levy'].str.replace('-', '0')
df['Levy'] = pd.to_numeric(df['Levy'], errors='coerce')
df['Mileage'] = pd.to_numeric(df['Mileage'].str.replace(' km', ''), errors='coerce')

df_original = df.copy()

df = df.dropna(subset=['Levy'])
df = df[(df['Cylinders'] >= 3) & (df['Cylinders'] <= 12)]
df = df[(df['Price'] >= 500)]

for col in ['Manufacturer', 'Category', 'Leather interior', 'Fuel type', 'Engine volume', 'Mileage',
            'Cylinders', 'Gear box type', 'Drive wheels', 'Doors', 'Turbo engine', 'Airbags']:
    missing_values = df[col].isnull().sum()

    if missing_values > 0:
        df = df.dropna(subset=[col])

cols_to_check = ['Price', 'Levy', 'Mileage']
for col in cols_to_check:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    factor = 4
    df = df[~((df[col] < (Q1 - factor * IQR)) | (df[col] > (Q3 + factor * IQR)))]
print(df.shape)
df_preEncoding = df.copy()

numeric_df = df.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numeric_df.corr()

# Generate the heatmap.
plt.figure(figsize=(12, 10))
sb.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix Heatmap')
plt.show()


df = pd.get_dummies(df, columns=['Manufacturer', 'Category', 'Fuel type', 'Leather interior', 'Gear box type',
                                 'Drive wheels', 'Doors'])
X = df.drop('Price', axis=1)
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=3)

#Three features reduction

features_to_plot = ['Mileage', 'Airbags', 'Engine volume']
colors = y

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Make sure 'df' contains all the data points corresponding to 'colors'
sc = ax.scatter(df[features_to_plot[0]],
                df[features_to_plot[1]],
                df[features_to_plot[2]],
                c=colors,  # This should be the same length as the features
                cmap='viridis')

# Color bar indicating price
cb = plt.colorbar(sc)
cb.set_label('Price')

# Labels for axes
ax.set_xlabel(features_to_plot[0])
ax.set_ylabel(features_to_plot[1])
ax.set_zlabel(features_to_plot[2])

plt.show()

#Data scaling
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#PCA dimension reduction
from sklearn.decomposition import PCA

# Perform PCA
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_train_scaled)

# Plotting the results
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y_train, cmap='viridis')

# Color bar indicating price
cb = plt.colorbar(sc)
cb.set_label('Price')

# Labels for axes
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')

plt.show()


from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(model, X_train, y_train, X_test, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)

    print(f"Train MSE: {mse_train}, Train R2: {r2_train}")
    print(f"Test MSE: {mse_test}, Test R2: {r2_test}")

    # Plotting the residuals for training and testing side by side
    fig, ax = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Training data residuals
    ax[0].scatter(y_train_pred, y_train_pred - y_train, c='blue', marker='o', label='Trénovacie dáta')
    ax[0].set_title('Reziduály trénovacích dát')
    ax[0].set_xlabel('Očakávané hodnoty')
    ax[0].set_ylabel('Reziduály')
    ax[0].hlines(y=0, xmin=min(y_train_pred), xmax=max(y_train_pred), color='red')
    ax[0].legend(loc='upper left')

    # Test data residuals
    ax[1].scatter(y_test_pred, y_test_pred - y_test, c='lightgreen', marker='s', label='Testovacie dáta')
    ax[1].set_title('Reziduály testovacích dát')
    ax[1].set_xlabel('Očakávané hodnoty')
    ax[1].hlines(y=0, xmin=min(y_test_pred), xmax=max(y_test_pred), color='red')
    ax[1].legend(loc='upper left')

    plt.show()

#Decision tree
from sklearn.tree import DecisionTreeRegressor, plot_tree
dt_model = DecisionTreeRegressor(max_depth=10,
                               min_samples_split=40,
                               min_samples_leaf=20,
                               max_features=None,
                               random_state=3)
dt_model.fit(X_train_scaled, y_train)
dt_model.fit(X_train_scaled, y_train)

# Visualize Decision Tree
plt.figure(figsize=(12, 8))
plot_tree(dt_model, filled=True, feature_names=X.columns, max_depth=10, fontsize=10)
plt.show()

evaluate_model(dt_model, X_train_scaled, y_train, X_test_scaled, y_test)

#Random forest
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(max_depth=10, random_state=3)
rf_model.fit(X_train_scaled, y_train)

importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

top_n = 15
plt.figure(figsize=(10, 15))
plt.title("Top 20 Feature Importances")
plt.barh(range(top_n), importances[indices][:top_n], align="center")
plt.yticks(range(top_n), X.columns[indices][:top_n])
plt.ylim([-1, top_n])
plt.tight_layout()
plt.show()

evaluate_model(rf_model, X_train_scaled, y_train, X_test_scaled, y_test)


#Support vector machine
from sklearn.svm import SVR
svm_model = SVR(kernel='poly', C=1000, gamma=0.5, epsilon=0.5)
svm_model.fit(X_train_scaled, y_train)
evaluate_model(svm_model, X_train_scaled, y_train, X_test_scaled, y_test)


#Random forests for subsets of features
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(max_depth=10, random_state=3)

#Based on corelation matrix

scaler = MinMaxScaler()
X_train_scaled_subset = scaler.fit_transform(X_train[['Prod. year', 'Engine volume', 'Levy']])
X_test_scaled_subset = scaler.transform(X_test[['Prod. year', 'Engine volume', 'Levy']])

X_train_heat = pd.DataFrame(X_train_scaled_subset, columns=['Prod. year', 'Engine volume', 'Levy'], index=X_train.index)
X_test_heat = pd.DataFrame(X_test_scaled_subset, columns=['Prod. year', 'Engine volume', 'Levy'], index=X_test.index)

rf_model.fit(X_train_heat, y_train)
evaluate_model(rf_model, X_train_heat, y_train, X_test_heat, y_test)

#Based on feature importance

X_train_scaled_importance = scaler.fit_transform(X_train[['Prod. year', 'Airbags', 'Mileage', 'Fuel type_Diesel', "Gear box type_Automatic", "Category_Jeep"]])
X_test_scaled_importance = scaler.transform(X_test[['Prod. year', 'Airbags', 'Mileage', 'Fuel type_Diesel', "Gear box type_Automatic", "Category_Jeep"]])

X_train_importance = pd.DataFrame(X_train_scaled_importance, columns=['Prod. year', 'Airbags', 'Mileage', 'Fuel type_Diesel', "Gear box type_Automatic", "Category_Jeep"], index=X_train.index)
X_test_importance = pd.DataFrame(X_test_scaled_importance, columns=['Prod. year', 'Airbags', 'Mileage', 'Fuel type_Diesel', "Gear box type_Automatic", "Category_Jeep"], index=X_test.index)

rf_model.fit(X_train_importance, y_train)
evaluate_model(rf_model, X_train_importance, y_train, X_test_importance, y_test)

#Based on PCA variancy

from sklearn.decomposition import PCA
pca = PCA(0.75)
pca.fit(X_train_scaled)
X_train_pca = pca.transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

rf_model.fit(X_train_pca, y_train)
evaluate_model(rf_model, X_train_pca, y_train, X_test_pca, y_test)