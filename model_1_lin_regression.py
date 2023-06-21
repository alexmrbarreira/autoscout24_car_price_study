from parameters import *

# This file fits the data with a linear regression model

# ===========================================================================
# Load data and encoders 
# ===========================================================================

df_train, df_valid, train_features, valid_features, train_labels, valid_labels, N_train, N_valid, N_featu = get_train_valid_data()
le_city, le_brand, le_body, le_year, le_gas, le_transmission, le_seller, le_owners, le_warranty = get_encoders()

# ===========================================================================
# Create, fit and save model
# ===========================================================================

# Fit model
model = linear_model.LinearRegression(fit_intercept=True)
model.fit(train_features, train_labels)

# Get predictions on training and validation sets
model_prediction_train = model.predict(train_features)
model_prediction_valid = model.predict(valid_features)

# Compute accuracy metric (mean squared error)
accuracy_train = np.mean( (model_prediction_train - train_labels)**2. )
accuracy_valid = np.mean( (model_prediction_valid - valid_labels)**2. )
print ('')
print ('Mean squared error:')
print ('On the training set:', accuracy_train)
print ('On the validation set:', accuracy_valid)

# Save model
pickle.dump(model, open('model_store/model_1_lin_regression.pickle', 'wb'))

# ===========================================================================
# Make accuracy plot
# ===========================================================================

fig0 = plt.figure(0, figsize=(17., 6.))
fig0.subplots_adjust(left = 0.065, right = 0.99, top = 0.93, bottom = 0.15, wspace = 0.20)

# Training performance
fig0.add_subplot(1,2,1)
plt.title('Training performance (lin. regression)', fontsize = title_font+4)
plt.scatter(train_labels, model_prediction_train, color = 'b', s = msize, marker = 'o')
xx   = np.linspace(minp_inplot, maxp_inplot, 10)
plt.plot(xx, xx, linewidth = 2, c = 'k', linestyle = 'dashed')
plt.xlim(minp_inplot, maxp_inplot)
plt.ylim(minp_inplot, maxp_inplot)
plt.xlabel('True price [1000 Eur]', fontsize = label_font+4)
plt.ylabel('Predicted price [1000 Eur]', fontsize = label_font+4)
plt.tick_params(length=tick_major, width=tickwidth, left=True, bottom=True, right=True, top=True, direction = 'in', which='major', pad=tickpad, labelsize = ticksize+10)
plt.annotate(r'Mean squared error: '+str(round(accuracy_train, 1)), xy = (0.10, 0.90), xycoords = 'axes fraction', fontsize = text_font, c = 'k')

# Validation performance
fig0.add_subplot(1,2,2)
plt.title('Validation performance (lin. regression)', fontsize = title_font+4)
plt.scatter(valid_labels, model_prediction_valid, color = 'b', s = msize, marker = 'o')
xx   = np.linspace(minp_inplot, maxp_inplot, 10)
plt.plot(xx, xx, linewidth = 2, c = 'k', linestyle = 'dashed')
plt.xlim(minp_inplot, maxp_inplot)
plt.ylim(minp_inplot, maxp_inplot)
plt.xlabel('True price [1000 Eur]', fontsize = label_font+4)
plt.ylabel('Predicted price [1000 Eur]', fontsize = label_font+4)
plt.tick_params(length=tick_major, width=tickwidth, left=True, bottom=True, right=True, top=True, direction = 'in', which='major', pad=tickpad, labelsize = ticksize+10)
plt.annotate(r'Mean squared error: '+str(round(accuracy_valid, 1)), xy = (0.10, 0.90), xycoords = 'axes fraction', fontsize = text_font, c = 'k')

fig0.savefig('fig_store/fig_model_1_lin_regression.png')

plt.show()
