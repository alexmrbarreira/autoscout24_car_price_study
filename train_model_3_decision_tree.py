from parameters import *

# This file fits the data with a decision tree regressor 

# ===========================================================================
# Load data and encoders 
# ===========================================================================

df_train, df_valid, train_features, valid_features, train_labels, valid_labels, N_train, N_valid, N_featu = get_train_valid_data()
le_city, le_brand, le_body, le_year, le_gas, le_transmission, le_seller, le_owners, le_warranty = get_encoders()

# ===========================================================================
# Create, fit and save model
# ===========================================================================

c_here = 'r' # color in plots below

# Fit model
model = tree.DecisionTreeRegressor(splitter='random')
model.fit(train_features, train_labels)

# Get predictions on training and validation sets
model_prediction_train = model.predict(train_features)
model_prediction_valid = model.predict(valid_features)

# Compute accuracy metrics
mse_train = np.mean( (model_prediction_train - train_labels)**2. )
mse_valid = np.mean( (model_prediction_valid - valid_labels)**2. )
print ('')
print ('Mean squared error:')
print ('On the training set:', mse_train)
print ('On the validation set:', mse_valid)
mpe_train = np.mean( abs( (model_prediction_train - train_labels)/train_labels ) )
mpe_valid = np.mean( abs( (model_prediction_valid - valid_labels)/valid_labels ) )
print ('')
print ('Mean percent error:')
print ('On the training set:', mpe_train)
print ('On the validation set:', mpe_valid)

# Save model
pickle.dump(model, open('model_store/model_3_decision_tree.pickle', 'wb'))

# ===========================================================================
# Make accuracy plot
# ===========================================================================

fig0 = plt.figure(0, figsize=(17., 6.))
fig0.subplots_adjust(left = 0.065, right = 0.99, top = 0.93, bottom = 0.15, wspace = 0.20)

# Training performance
fig0.add_subplot(1,2,1)
plt.title('Training performance (decision tree)', fontsize = title_font+4)
plt.scatter(train_labels, model_prediction_train, color = c_here, s = msize, marker = 'o')
xx   = np.linspace(minp_inplot, maxp_inplot, 10)
plt.plot(xx, xx, linewidth = 2, c = 'k', linestyle = 'dashed')
plt.xlim(minp_inplot, maxp_inplot)
plt.ylim(minp_inplot, maxp_inplot)
plt.xlabel('True price [1000 Eur]', fontsize = label_font+4)
plt.ylabel('Predicted price [1000 Eur]', fontsize = label_font+4)
plt.tick_params(length=tick_major, width=tickwidth, left=True, bottom=True, right=True, top=True, direction = 'in', which='major', pad=tickpad, labelsize = ticksize+10)
plt.annotate(r'Mean squared error: '+str(round(mse_train, 1)), xy = (0.10, 0.90), xycoords = 'axes fraction', fontsize = text_font, c = c_here)
plt.annotate(r'Mean percent error: '+str(round(mpe_train, 2)), xy = (0.10, 0.83), xycoords = 'axes fraction', fontsize = text_font, c = c_here)

# Validation performance
fig0.add_subplot(1,2,2)
plt.title('Validation performance (decision tree)', fontsize = title_font+4)
plt.scatter(valid_labels, model_prediction_valid, color = c_here, s = msize, marker = 'o')
xx   = np.linspace(minp_inplot, maxp_inplot, 10)
plt.plot(xx, xx, linewidth = 2, c = 'k', linestyle = 'dashed')
plt.xlim(minp_inplot, maxp_inplot)
plt.ylim(minp_inplot, maxp_inplot)
plt.xlabel('True price [1000 Eur]', fontsize = label_font+4)
plt.ylabel('Predicted price [1000 Eur]', fontsize = label_font+4)
plt.tick_params(length=tick_major, width=tickwidth, left=True, bottom=True, right=True, top=True, direction = 'in', which='major', pad=tickpad, labelsize = ticksize+10)
plt.annotate(r'Mean squared error: '+str(round(mse_valid, 1)), xy = (0.10, 0.90), xycoords = 'axes fraction', fontsize = text_font, c = c_here)
plt.annotate(r'Mean percent error: '+str(round(mpe_valid, 2)), xy = (0.10, 0.83), xycoords = 'axes fraction', fontsize = text_font, c = c_here)

fig0.savefig('fig_store/fig_model_3_decision_tree.png')

plt.show()
