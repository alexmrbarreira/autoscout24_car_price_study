from parameters import *

# This file fits the data with a linear regression model

# ===========================================================================
# Load data and encoders 
# ===========================================================================

df_train, df_valid, train_features, valid_features, train_labels, valid_labels, N_train, N_valid, N_featu = get_train_valid_data()
le_city, le_brand, le_body, le_year, le_gas, le_transmission, le_seller, le_owners, le_warranty = get_encoders()

# ===========================================================================
# Create and fit model
# ===========================================================================

#model = linear_model.LinearRegression()
model = neighbors.KNeighborsRegressor(n_neighbors = 2)
#model = tree.DecisionTreeRegressor(splitter='random')

model.fit(train_features, train_labels)

#model_prediction = model.predict(train_features)
model_prediction = model.predict(valid_features)

#plt.scatter(model_prediction, train_labels, s = 5)
plt.scatter(model_prediction, valid_labels, s = 5)
plt.plot(range(0, 100), range(0, 100), linewidth = 3)
plt.show()

