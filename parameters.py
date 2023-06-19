from bs4 import BeautifulSoup, SoupStrainer #HTML parsing
import numpy as np
import matplotlib.pyplot as plt
import requests
import os
import pandas as pd
pd.options.mode.chained_assignment = None
from sklearn import preprocessing
from pickle import dump, load
plt.rcParams.update({'text.usetex': True, 'mathtext.fontset': 'stix'}) #['dejavuserif', 'cm', 'custom', 'stix', 'stixsans', 'dejavusans']

# ======================================================= 
# Search parameters
# ======================================================= 
country   = 'D'
max_page  = 20
min_year  = '2013'

# Body types
body_names = ['kleinwagen', 'cabrio', 'coupe', 'suv', 'kombi', 'limousine', 'van', 'transporter', 'sonstige']
body_types = [    '1'     ,                             '5'  ,      '6'   ]

# Car brands (could also refine search by loading only certain models per brand)
#brand_list = ['audi', 'bmw', 'ford', 'mercedes', 'opel', 'skoda', 'toyota', 'volkswagen', 'volvo']
brand_list = [         'bmw',                             'skoda',           'volkswagen']

# The autoscout24 website does not show more than 20 pages (and ~20 cars per page, so search is complete only if filters result in about 400 cars)
# To circumvent this, the search loops over sufficiently small price bins that there are never > 400 cars inside each; this guarantees we catch practically all cars
price_bin_edges = ['500', '5000', '10000', '15000', '20000', '25000', '30000', '40000', '50000', '75000', '100000']

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

city_list = [ ['münchen'    , 'lat=48.13913&lon=11.58022'] ,
              ['berlin'     , 'lat=52.52343&lon=13.41144'] ]

# ======================================================= 
# Function that load stuff
# ======================================================= 

# Function that load training and validation data
def get_train_valid_data():
    # Read files
    df_train = pd.read_csv('data_store/data_prepared_train.csv')
    df_valid = pd.read_csv('data_store/data_prepared_valid.csv')
    # Get sizes
    N_featu = len(df_train.columns.tolist())
    N_train = df_train.shape[0]
    N_valid = df_valid.shape[0]
    # Get features
    train_features = df_train.drop(['Price'], axis = 1)
    valid_features = df_valid.drop(['Price'], axis = 1)
    # Get labels
    train_labels = df_train['Price']
    valid_labels = df_valid['Price']
    # return 
    return df_train, df_valid, train_features, valid_features, train_labels, valid_labels, N_train, N_valid, N_featu

def get_encoders():
    le_city         = load(open('encoder_store/le_city.pkl', 'rb'))
    le_brand        = load(open('encoder_store/le_brand.pkl', 'rb'))
    le_body         = load(open('encoder_store/le_body.pkl', 'rb'))
    le_gas          = load(open('encoder_store/le_gas.pkl', 'rb'))
    le_transmission = load(open('encoder_store/le_transmission.pkl', 'rb'))
    le_seller       = load(open('encoder_store/le_seller.pkl', 'rb'))
    le_warranty     = load(open('encoder_store/le_warranty.pkl', 'rb'))
    return le_city, le_brand, le_body, le_gas, le_transmission, le_seller, le_warranty

# ======================================================= 
# Basic ploting parameters 
# ======================================================= 
ticksize    = 22
tick_major  = 10.
tick_minor  = 5.
tickwidth   = 1.5
label_font  = 22
title_font  = 22
text_font   = 22
legend_font = 22
tickpad     = 6.
alpha_c     = 0.3



