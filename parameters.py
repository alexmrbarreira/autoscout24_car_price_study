from bs4 import BeautifulSoup, SoupStrainer #HTML parsing
import numpy as np
import matplotlib.pyplot as plt
import requests
import os
import pandas as pd
pd.options.mode.chained_assignment = None
from sklearn import preprocessing, linear_model, neighbors, tree, neural_network, ensemble
import pickle
plt.rcParams.update({'text.usetex': True, 'mathtext.fontset': 'stix'}) #['dejavuserif', 'cm', 'custom', 'stix', 'stixsans', 'dejavusans']

# ======================================================= 
# Search parameters
# ======================================================= 
country   = 'D'
max_page  = 20
min_year  = '2013'

# Body types
body_names = ['kleinwagen', 'cabrio', 'coupe', 'suv', 'kombi', 'limousine', 'van', 'transporter', 'sonstige']
#body_types = [    '1'     ,    '2'  ,   '3'  ,  '4' ,   '5'  ,      '6'   ,  '7' ,      '8',    ,     '9'   ]
body_types = [    '1'     ,                     '4' ,   '5'  ,      '6'   ]

# Car brands 
brand_list = ['audi', 'bmw', 'ford', 'mercedes-benz', 'opel', 'skoda', 'toyota', 'volkswagen', 'volvo']

# German state capitals
search_radius = '100' #1000 x km
#city_list = [ ['münchen'    , 'lat=48.13913&lon=11.58022'],
#              ['stuttgart'  , 'lat=48.77711&lon=9.18077' ],
#              ['düsseldorf' , 'lat=51.22496&lon=6.77568' ],
#              ['dresden'    , 'lat=51.05099&lon=13.73363'],
#              ['hannover'   , 'lat=52.37207&lon=9.73569' ], 
#              ['kiel'       , 'lat=54.32276&lon=10.1359' ],
#              ['erfurt'     , 'lat=50.97374&lon=11.02243'], 
#              ['wiesbaden'  , 'lat=50.08406&lon=8.2398'  ], 
#              ['mainz'      , 'lat=49.99511&lon=8.26739' ], 
#              ['saarbrücken', 'lat=49.23478&lon=6.9944'  ], 
#              ['schwerin'   , 'lat=53.62574&lon=11.41689'], 
#              ['magdeburg'  , 'lat=52.13167&lon=11.64032'], 
#              ['potsdam'    , 'lat=52.39615&lon=13.05854'], 
#              ['bremen'     , 'lat=53.07498&lon=8.80708' ], 
#              ['berlin'     , 'lat=52.52343&lon=13.41144'],
#              ['hamburg'    , 'lat=53.55725&lon=9.99597' ] ]

city_list = [ ['hamburg'    , 'lat=53.55725&lon=9.99597' ] ]
              


# The autoscout24 website does not show more than 20 pages (and ~20 cars per page, so search is complete only if filters result in about 400 cars).
# To circumvent this, the search loops over sufficiently small price bins that there are never > 400 cars inside each.
# This catches all car prices; modify this only if want a restricted price range (though that selection can be done at post-processing).
price_bin_edges = ['500', '5000', '10000', '15000', '20000', '25000', '30000', '40000', '50000', '75000', '100000']

# ======================================================= 
# Function that load stuff
# ======================================================= 

# Function that load training and validation data
def get_train_valid_data():
    # Read files
    df_train = pd.read_csv('data_store/data_prepared_train.csv')
    df_valid = pd.read_csv('data_store/data_prepared_valid.csv')
    # Get sizes
    N_featu = len(df_train.columns.tolist()) - 1
    N_train = df_train.shape[0]
    N_valid = df_valid.shape[0]
    # Get features
    train_features = df_train.drop(['Price[1000Eur]'], axis = 1).to_numpy()
    valid_features = df_valid.drop(['Price[1000Eur]'], axis = 1).to_numpy()
    # Get labels
    train_labels = df_train['Price[1000Eur]'].to_numpy()
    valid_labels = df_valid['Price[1000Eur]'].to_numpy()
    # return 
    return df_train, df_valid, train_features, valid_features, train_labels, valid_labels, N_train, N_valid, N_featu

# Function that loads the encoders for inverse_transformations in plots
def get_encoders():
    le_city         = pickle.load(open('encoder_store/le_city.pkl', 'rb'))
    le_brand        = pickle.load(open('encoder_store/le_brand.pkl', 'rb'))
    le_body         = pickle.load(open('encoder_store/le_body.pkl', 'rb'))
    le_year         = pickle.load(open('encoder_store/le_year.pkl', 'rb'))
    le_gas          = pickle.load(open('encoder_store/le_gas.pkl', 'rb'))
    le_transmission = pickle.load(open('encoder_store/le_transmission.pkl', 'rb'))
    le_seller       = pickle.load(open('encoder_store/le_seller.pkl', 'rb'))
    le_owners       = pickle.load(open('encoder_store/le_owners.pkl', 'rb'))
    le_warranty     = pickle.load(open('encoder_store/le_warranty.pkl', 'rb'))
    return le_city, le_brand, le_body, le_year, le_gas, le_transmission, le_seller, le_owners, le_warranty

## Function that scales the features and labels by their maxima
#def scale_data(train_features, valid_features, train_labels, valid_labels):
#    new_train_features = np.zeros(np.shape(train_features))
#    new_valid_features = np.zeros(np.shape(valid_features))
#    new_train_labels = np.zeros(np.shape(train_labels))
#    new_valid_labels = np.zeros(np.shape(valid_labels))
#    N_featu = len(train_features[0,:])
#    for i in range(N_featu):
#        new_train_features[:,i] = train_features[:,i]/max(train_features[:,i])
#        new_valid_features[:,i] = valid_features[:,i]/max(valid_features[:,i])
#    new_train_labels = train_labels/max(train_labels)
#    new_valid_labels = valid_labels/max(valid_labels)
#    return new_train_features, new_valid_features, new_train_labels, new_valid_labels

# ======================================================= 
# Basic ploting parameters 
# ======================================================= 
ticksize    = 16
tick_major  = 10.
tick_minor  = 5.
tickwidth   = 1.5
label_font  = 22
title_font  = 22
text_font   = 22
legend_font = 22
tickpad     = 6.
alpha_c     = 0.3
msize       = 0.02

minp_inplot = -10
maxp_inplot = 110

