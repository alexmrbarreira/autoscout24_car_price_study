from parameters import *

#verbose = True
verbose = False

# ================================================================ 
# Load data 
# ================================================================ 

df_train, df_valid, train_features, valid_features, train_labels, valid_labels, N_train, N_valid, N_featu = get_train_valid_data()

#df_total = pd.concat([df_train, df_valid])
#df_touse = df_total.drop(['Price[1000Eur]'], axis=1)
#df_touse = df_train.drop(['Price[1000Eur]'], axis=1)
df_touse = df_valid.drop(['Price[1000Eur]'], axis=1)
cols     = df_touse.columns.tolist()

le_city, le_brand, le_body, le_year, le_gas, le_transmission, le_seller, le_owners, le_warranty = get_encoders()
list_of_le = [le_city, le_brand, le_body, None, None, le_year, le_gas, le_transmission, le_seller, le_owners, le_warranty]

# ================================================================ 
# Load models and choose one to do the analysis with
# ================================================================ 

model_1 = pickle.load(open('model_store/model_1_lin_regression.pickle'   , 'rb'))
model_2 = pickle.load(open('model_store/model_2_kNearestNeighbors.pickle', 'rb'))
model_3 = pickle.load(open('model_store/model_3_decision_tree.pickle'    , 'rb'))
model_4 = pickle.load(open('model_store/model_4_random_forest.pickle'    , 'rb'))
model_5 = pickle.load(open('model_store/model_5_MLperceptron.pickle'     , 'rb'))

model_list  = [     model_1     ,       model_2      ,      model_3   ,      model_4   ,         model_5      ]
model_names = ['Lin. regression', 'k near neighbors' , 'Decision tree', 'Random forest', 'Multi-layer percep.']
model_c     = [      'b'        ,         'g'        ,       'r'      ,   'darkorange' ,            'c'       ]

imodel = 3
print ('')
print ('Quantifying feature impact with model:', model_names[imodel])
print ('')
model = model_list[imodel]

# ================================================================
# Quantify impact of each feature 
# ================================================================

# The quantification of a feature impact is done by checking the impact on mean 
# car prices that don't have the feature if they are given that feature.
# E.g. What's the percentage change in prices of all manual cars, if they are assumed automatic

#   0        1       2        3         4           5       6          7            8          9         10
#['City', 'Brand', 'Body', '1000Km', 'Power[HP]', 'Year', 'Gas', 'Transmission', 'Seller', 'Owners', 'Warranty']

#Label encoding of City: ['berlin' 'münchen'] ----> [0 1]
#Label encoding of Brand: ['audi' 'bmw' 'ford' 'mercedes' 'opel' 'skoda' 'toyota' 'volkswagen' 'volvo'] ----> [0 1 2 3 4 5 6 7 8]
#Label encoding of Body: ['kleinwagen' 'kombi' 'limousine' 'suv'] ----> [0 1 2 3]
#Label encoding of Year: [2013 2014 2015 2016 2017 2018 2019 2020 2021 2022 2023] ----> [ 0  1  2  3  4  5  6  7  8  9 10]
#Label encoding of Gas: ['Benzin' 'Diesel' 'Other'] ----> [0 1 2]
#Label encoding of Transmission: ['Automatik' 'Schaltgetriebe'] ----> [0 1]
#Label encoding of Seller: ['Händler' 'Privat'] ----> [0 1]
#Label encoding of Owners: [1. 2. 3. 4. 5. 7.] ----> [0 1 2 3 4 5]
#Label encoding of Warranty: ['Ja' 'Ja-exp' 'Nein'] ----> [0 1 2]

# Type of category of each feature
category_type = ['category', 'category', 'category', 'bins', 'bins', 'category', 'category', 'category', 'category', 'category', 'category']

# Build bins for '1000km'
nbins        = 15 # for non-categorial variables
min_val      = 20.
max_val      = 200.
bin_edges_3  = np.linspace(min_val, max_val, nbins+1)
bin_means_3  = (bin_edges_3[1::] + bin_edges_3[0:-1])/2.

# Build bins for 'Power[HP]'
nbins        = 15 # for non-categorial variables
min_val      = 75.
max_val      = 250.
bin_edges_4  = np.linspace(min_val, max_val, nbins+1)
bin_means_4  = (bin_edges_4[1::] + bin_edges_4[0:-1])/2.

list_of_bin_edges = [None, None, None, bin_edges_3, bin_edges_4, None, None, None, None, None, None]
list_of_bin_means = [None, None, None, bin_means_3, bin_means_4, None, None, None, None, None, None]

# Loop over over features
feature_impact = []
for i in range(len(cols)):
    feature = cols[i]
    if (verbose):
        print ('')
        print ('==========================================================================================')
        print ('******************************************************************************************')
        print ('==========================================================================================')
    print ('feature = ', feature)

    feature_impacts_now = []
    # Deal with categorial variables
    if(category_type[i] == 'category'):
        le_now             = list_of_le[i]
        encoded_variables  = le_now.transform(le_now.classes_)

        # Loop over categories in feature
        for j in range(len(le_now.classes_)):
            if (verbose):
                print ('')
                print ('Category now:', le_now.classes_[j], encoded_variables[j])

            # Get predicted prices for all cars not in this category
            df_now             = df_touse.loc[df_touse[feature] != encoded_variables[j]]
            prediction_def_now = model.predict(df_now.values)

            if (verbose):
                print ('')
                print ('DF for all cars not in this category')
                print (df_now)
            
            # Get predicted prices assuming all of the above cars now have this catogory
            df_now[feature]    = encoded_variables[j]
            prediction_mod_now = model.predict(df_now.values)

            if (verbose):
                print ('')
                print ('DF for all cars not in this category, assuming they get this category')
                print (df_now)

            # Estimate impact (mean percentage change)
            feature_impacts_now.append( 100. * np.mean(prediction_mod_now/prediction_def_now-1.) )

    # Deal with non-categorial variables
    else:
        bin_edges = list_of_bin_edges[i]
        bin_means = list_of_bin_means[i]
        # Loop over bins in feature
        for j in range(len(bin_edges)-1):

            if (verbose):
                print ('')
                print ('Bin now:', bin_edges[j], bin_edges[j+1], bin_means[j])

            # Get predicted prices for all cars not in this bin
            df_now             = df_touse.loc[ (df_touse[feature] < bin_edges[j]) | (df_touse[feature] > bin_edges[j+1]) ]
            prediction_def_now = model.predict(df_now.values)

            if (verbose):
                print ('')
                print ('DF for all cars not in this bin')
                print (df_now)

            # Get predicted prices assuming all of the above cars have now this bin's mean value
            df_now[feature]    = bin_means[j]
            prediction_mod_now = model.predict(df_now.values)

            if (verbose):
                print ('')
                print ('DF for all cars not in this bin, assuming they get this bin mean')
                print (df_now)

            # Estimate impact (mean percentage change)
            feature_impacts_now.append( 100. * np.mean(prediction_mod_now/prediction_def_now-1.) )

    feature_impact.append( feature_impacts_now )

if (verbose):
    print ('The feature impacts [%]:')
    for i in range(len(cols)):
        print (cols[i])
        print ([round(a,2) for a in feature_impact[i]])

# ================================================================
# Make plot 
# ================================================================

fig0 = plt.figure(0, figsize = (17., 10.))
fig0.subplots_adjust(left=0.06, right=0.99, bottom=0.10, top = 0.98, hspace = 0.45, wspace = 0.2)

#adaptive_max = True
adaptive_max = False
nonadap_max  = 25.

for i in range(len(cols)):
    fig0.add_subplot(3, 4 , i+2)

    feature = cols[i]

    # Deal with categorial variables
    if(category_type[i] == 'category'):
        le_now = list_of_le[i]
        xx = range(len(le_now.classes_))
        yy = feature_impact[i]
        # Color according to <0 or >0
        cc = np.array(['b' for i in range(len(yy))])
        cc[np.where(np.array(yy) > 0)[0]] = 'r'
        plt.bar(xx, feature_impact[i], color = cc, alpha = alpha_c)
        # Cosmetics
        if( (feature=='Brand') or (feature=='Year') ):
            plt.xticks(xx, le_now.classes_, rotation = 55)
        else:
            plt.xticks(xx, le_now.classes_, rotation = 30)

    # Deal with non-categorical variables
    else:
        xx = list_of_bin_means[i]
        yy = feature_impact[i]
        xx_width = xx[1]-xx[0]
        # Color according to <0 or >0
        cc = np.array(['b' for i in range(len(yy))])
        cc[np.where(np.array(yy) > 0)[0]] = 'r'
        plt.bar(xx, feature_impact[i], color = cc, alpha = alpha_c, width = (9./10.)*xx_width)
        plt.xlabel(cols[i], fontsize = label_font)

    # Other cosmetics
    if(adaptive_max):
        max_abs_val = max(abs(np.array(feature_impact[i]))) * 1.10
        plt.ylim(-max_abs_val, max_abs_val)
    else:
        plt.ylim(-nonadap_max, nonadap_max)
    plt.axhline(0., linestyle = 'dashed', c = 'k', linewidth = 2.)
    if ( (i==0) or (i==3) or (i==7) ):
        plt.ylabel(r'Feature impact [\%]', fontsize = label_font)
    plt.tick_params(length=tick_major, width=tickwidth , bottom=True, top=True, left=True, right=True, direction = 'in', which = 'major', pad = tickpad, labelsize = ticksize)
    if ( (i==9) or (i==10) ):
        plt.xlabel(cols[i], fontsize = label_font)
    if (i==0):
        plt.annotate('Results from model: \n '+model_names[imodel], xy = (-1.25, 0.3), xycoords = 'axes fraction', fontsize = text_font+8, c = model_c[imodel])

fig0.savefig('fig_store/fig_feature_impact_single_model.png')

plt.show()





