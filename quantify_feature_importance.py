from parameters import *

# ================================================================ 
# Load data 
# ================================================================ 

df_train, df_valid, train_features, valid_features, train_labels, valid_labels, N_train, N_valid, N_featu = get_train_valid_data()

cols     = df_train.drop(['Price[1000Eur]'], axis=1).columns.tolist()

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

# Tree impurity-based feature importances for comparison
fi_3 = model_3.feature_importances_
fi_4 = model_4.feature_importances_

# ================================================================
# Estimate feature importance by randomization 
# ================================================================

def get_feature_importance(model, data_features, data_labels):
    N_data     = len(data_features[:,0])
    N_features = len(data_features[0,:])
    # Default model prediction and accuracy
    prediction_default = model.predict(data_features)
    accuracy_default   = np.mean( abs((prediction_default - data_labels)/data_labels) )
    # Measure feature importance by size of accuracy loss after randomization
    feature_importance = np.zeros(N_features)
    for j in range(N_features):
        data_features_now = np.copy(data_features)
        np.random.shuffle(data_features_now[:,j])
        prediction_now = model.predict(data_features_now)
        accuracy_now   = np.mean( abs((prediction_now - data_labels)/data_labels) ) 
        feature_importance[j] = accuracy_default/accuracy_now
    return feature_importance, accuracy_default

def get_average_feature_importance(model, data_features, data_labels, N_random):
    N_features                 = len(data_features[0,:])
    average_feature_importance = np.zeros(N_features)
    for i in range(N_random):
        average_feature_importance += get_feature_importance(model, data_features, data_labels)[0]
    return average_feature_importance/N_random, get_feature_importance(model, data_features, data_labels)[1]

N_random = 1
feature_importance_train_list = []
feature_importance_valid_list = []
for i in range(Nmodels):
    print ('Estimating feature importance by randomization for', model_names[i])
    feature_importance_train_list.append( get_average_feature_importance(model_list[i], train_features, train_labels, N_random)[0] )
    feature_importance_valid_list.append( get_average_feature_importance(model_list[i], valid_features, valid_labels, N_random)[0] )

# ================================================================
# Make feature importance plot
# ================================================================

fig0 = plt.figure(0, figsize=(19., 12.))
fig0.subplots_adjust(left=0.04, bottom=0.11, right=0.99, top=0.96, wspace=0.25, hspace=0.40)

msize_here = 20.

args_order = np.array([4,5,3,6,7,1,2,10,0,9,8])

# Add training feature importances in the upper panels
for i in range(Nmodels):
    fig0.add_subplot(2, 5, i+1)
    xx = range(N_featu)
    plt.scatter(xx, 1./feature_importance_train_list[i][args_order], color = model_c[i], s = msize_here, marker = 'o')
    plt.plot(xx, 1./feature_importance_train_list[i][args_order], linewidth = 2, c = model_c[i], linestyle = 'dashed')
    plt.xticks(xx, [cols[i] for i in args_order], fontsize = label_font-10, rotation=90.)
    plt.tick_params(length=tick_major, width=tickwidth, left=True, bottom=True, right=True, top=True, direction = 'in', which='major', pad=tickpad, labelsize = ticksize)
    plt.annotate(model_names[i]                           , xy = (0.20, 0.82), xycoords = 'axes fraction', fontsize = text_font-2, c = model_c[i])
    if(i==0): plt.ylabel('1/accuracy loss by randomization', fontsize = label_font)
    if(i==2): plt.title('Training feature importances', fontsize = title_font+4)
    # for the tree based regressors, compare also with the impurity-based estimates
    if(i==2): 
        normalized_inpurity_based_feature_importance = (max(1./feature_importance_train_list[i][args_order])-1)*fi_3[args_order]/max(fi_3[args_order]) + 1.
        plt.plot(xx, normalized_inpurity_based_feature_importance, linewidth = 2., linestyle = 'dotted', c = 'k', label = 'Inpurity-\nbased')
        params = {'legend.fontsize': legend_font-4}; plt.rcParams.update(params); plt.legend(loc = 'center right', ncol = 1)
    if(i==3):
        normalized_inpurity_based_feature_importance = (max(1./feature_importance_train_list[i][args_order])-1)*fi_4[args_order]/max(fi_4[args_order]) + 1.
        plt.plot(xx, normalized_inpurity_based_feature_importance, linewidth = 2., linestyle = 'dotted', c = 'k', label = 'Inpurity-\nbased')
        params = {'legend.fontsize': legend_font-4}; plt.rcParams.update(params); plt.legend(loc = 'center right', ncol = 1)


# Add validing performance in the lower panels
for i in range(Nmodels):
    fig0.add_subplot(2, 5, i+1+5)
    xx = range(N_featu)
    plt.scatter(xx, 1./feature_importance_valid_list[i][args_order], color = model_c[i], s = msize_here, marker = 'o')
    plt.plot(xx, 1./feature_importance_valid_list[i][args_order], linewidth = 2, c = model_c[i], linestyle = 'dashed')
    plt.xticks(xx, [cols[j] for j in args_order], fontsize = label_font-10, rotation=90.)
    plt.tick_params(length=tick_major, width=tickwidth, left=True, bottom=True, right=True, top=True, direction = 'in', which='major', pad=tickpad, labelsize = ticksize)
    plt.annotate(model_names[i]                           , xy = (0.20, 0.82), xycoords = 'axes fraction', fontsize = text_font-2, c = model_c[i])
    if(i==0): plt.ylabel('1/accuracy loss by randomization', fontsize = label_font)
    if(i==2): plt.title('Validation feature importances', fontsize = title_font+4)
    # for the tree based regressors, compare also with the impurity-based estimates
    if(i==2): 
        normalized_inpurity_based_feature_importance = (max(1./feature_importance_valid_list[i][args_order])-1)*fi_3[args_order]/max(fi_3[args_order]) + 1.
        plt.plot(xx, normalized_inpurity_based_feature_importance, linewidth = 2., linestyle = 'dotted', c = 'k', label = 'Inpurity-\nbased')
        params = {'legend.fontsize': legend_font-4}; plt.rcParams.update(params); plt.legend(loc = 'center right', ncol = 1)
    if(i==3):
        normalized_inpurity_based_feature_importance = (max(1./feature_importance_valid_list[i][args_order])-1)*fi_4[args_order]/max(fi_4[args_order]) + 1.
        plt.plot(xx, normalized_inpurity_based_feature_importance, linewidth = 2., linestyle = 'dotted', c = 'k', label = 'Inpurity-\nbased')
        params = {'legend.fontsize': legend_font-4}; plt.rcParams.update(params); plt.legend(loc = 'center right', ncol = 1)

fig0.savefig('fig_store/fig_feature_importances_by_randomization.png')

# ================================================================
# Make feature importance plot just for the random forest (best)
# ================================================================

fig1 = plt.figure(1, figsize=(11., 9.))
fig1.subplots_adjust(left=0.08, bottom=0.16, right=0.99, top=0.94, wspace=0.25, hspace=0.40)

i = 3
fig1.add_subplot(1, 1, 1)
xx = range(N_featu)
plt.title('Which car features play a dominant role in the price?', fontsize = title_font+4)
plt.scatter(xx, 1./feature_importance_valid_list[i][args_order], color = model_c[i], s = 200, marker = 'o')
plt.plot(xx, 1./feature_importance_valid_list[i][args_order], linewidth = 2, c = model_c[i], linestyle = 'dashed')
plt.xticks(xx, [cols[i] for i in args_order], fontsize = label_font, rotation=45.)
plt.tick_params(length=tick_major, width=tickwidth, left=True, bottom=True, right=True, top=True, direction = 'in', which='major', pad=tickpad, labelsize = ticksize+6)
plt.ylabel('Car feature importance', fontsize = label_font+6)
plt.annotate('Results from model: \n '+model_names[i], xy = (0.4, 0.65), xycoords = 'axes fraction', fontsize = text_font+8, c = model_c[i])

fig1.savefig('fig_store/fig_feature_importances_by_randomization_model_4_random_forest.png')

plt.show()

