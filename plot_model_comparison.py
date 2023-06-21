from parameters import *

# ================================================================ 
# Load data 
# ================================================================ 

df_train, df_valid, train_features, valid_features, train_labels, valid_labels, N_train, N_valid, N_featu = get_train_valid_data()

df_total = pd.concat([df_train, df_valid])
cols     = df_total.columns.tolist()

# ================================================================ 
# Load models 
# ================================================================ 

model_1 = pickle.load(open('model_store/model_1_lin_regression.pickle'   , 'rb'))
model_2 = pickle.load(open('model_store/model_2_kNearestNeighbors.pickle', 'rb'))
model_3 = pickle.load(open('model_store/model_3_decision_tree.pickle'    , 'rb'))
model_4 = pickle.load(open('model_store/model_4_random_forest.pickle'    , 'rb'))
model_5 = pickle.load(open('model_store/model_5_MLperceptron.pickle'     , 'rb'))

model_list  = [     model_1     ,       model_2      ,      model_3   ,      model_4   ,         model_5      ]
model_names = ['Lin. regression', 'k near neighbors' , 'Decision tree', 'Random forest', 'Multi-layer percep.']
model_c     = [      'b'        ,         'g'        ,       'r'      ,   'darkorange' ,            'c'       ]
Nmodels     = len(model_list)

# ================================================================
# Compute model performance metrics 
# ================================================================

# model predictions
model_prediction_train_list = []
model_prediction_valid_list = []
for model in model_list:
    model_prediction_train_list.append( model.predict(train_features) )
    model_prediction_valid_list.append( model.predict(valid_features) )

# mean squared errors
mse_train_list = []
mse_valid_list = []
for i in range(Nmodels):
    mse_train_list.append( np.mean( (model_prediction_train_list[i] - train_labels)**2. ) )
    mse_valid_list.append( np.mean( (model_prediction_valid_list[i] - valid_labels)**2. ) )

# mean percent errors
mpe_train_list = []
mpe_valid_list = []
for i in range(Nmodels):
    mpe_train_list.append( np.mean( abs((model_prediction_train_list[i] - train_labels)/train_labels) ) )
    mpe_valid_list.append( np.mean( abs((model_prediction_valid_list[i] - valid_labels)/valid_labels) ) )

# ================================================================
# Make predicted vs. true plot
# ================================================================

fig0 = plt.figure(0, figsize=(17., 9.))
fig0.subplots_adjust(left=0.06, bottom=0.10, right=0.99, top=0.94, wspace=0.25, hspace=0.30)

# Add training performance in the upper panels
for i in range(Nmodels):
    fig0.add_subplot(2, 5, i+1)
    plt.scatter(train_labels, model_prediction_train_list[i], color = model_c[i], s = msize, marker = 'o')
    xx   = np.linspace(minp_inplot, maxp_inplot, 10)
    plt.plot(xx, xx, linewidth = 2, c = 'k', linestyle = 'dashed')
    plt.xlim(minp_inplot, maxp_inplot)
    plt.ylim(minp_inplot, maxp_inplot)
    plt.tick_params(length=tick_major, width=tickwidth, left=True, bottom=True, right=True, top=True, direction = 'in', which='major', pad=tickpad, labelsize = ticksize+10)
    plt.annotate(model_names[i]                           , xy = (0.05, 0.82), xycoords = 'axes fraction', fontsize = text_font-2, c = model_c[i])
    plt.annotate(r'MSE: '+str(round(mse_train_list[i], 1)), xy = (0.50, 0.15), xycoords = 'axes fraction', fontsize = text_font-6, c = model_c[i])
    plt.annotate(r'MPE: '+str(round(mpe_train_list[i], 2)), xy = (0.50, 0.08), xycoords = 'axes fraction', fontsize = text_font-6, c = model_c[i])
    if(i==0): plt.ylabel('Predicted price [1000 Eur]', fontsize = label_font)
    if(i==2): plt.title('Training performance', fontsize = title_font+4)

# Add validation performance in the lower panels
for i in range(Nmodels):
    fig0.add_subplot(2, 5, i+1+5)
    plt.scatter(valid_labels, model_prediction_valid_list[i], color = model_c[i], s = msize, marker = 'o')
    xx   = np.linspace(minp_inplot, maxp_inplot, 10)
    plt.plot(xx, xx, linewidth = 2, c = 'k', linestyle = 'dashed')
    plt.xlim(minp_inplot, maxp_inplot)
    plt.ylim(minp_inplot, maxp_inplot)
    plt.xlabel('True price [1000 Eur]', fontsize = label_font)
    plt.tick_params(length=tick_major, width=tickwidth, left=True, bottom=True, right=True, top=True, direction = 'in', which='major', pad=tickpad, labelsize = ticksize+10)
    plt.annotate(model_names[i]                           , xy = (0.05, 0.82), xycoords = 'axes fraction', fontsize = text_font-2, c = model_c[i])
    plt.annotate(r'MSE: '+str(round(mse_valid_list[i], 1)), xy = (0.50, 0.15), xycoords = 'axes fraction', fontsize = text_font-6, c = model_c[i])
    plt.annotate(r'MPE: '+str(round(mpe_valid_list[i], 2)), xy = (0.50, 0.08), xycoords = 'axes fraction', fontsize = text_font-6, c = model_c[i])
    if(i==0): plt.ylabel('Predicted price [1000 Eur]', fontsize = label_font)
    if(i==2): plt.title('Validation performance', fontsize = title_font+4)


fig0.savefig('fig_store/fig_model_comparison.png')

plt.show()



















