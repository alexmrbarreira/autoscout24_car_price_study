from bs4 import BeautifulSoup, SoupStrainer #HTML parsing
import numpy as np
import matplotlib.pyplot as plt
import requests
import os
import pandas as pd
plt.rcParams.update({'text.usetex': True, 'mathtext.fontset': 'stix'}) #['dejavuserif', 'cm', 'custom', 'stix', 'stixsans', 'dejavusans']

# ======================================================= 
# Search parameters
# ======================================================= 
country   = 'D'
max_page  = 20 
min_year  = '2013'

# Body types
#            [small car, suv, station wagon, sedan]
#body_types = [   '1'   , '4',       '5'    ,  '6' ]
body_types = [                       '5'    ,  '6' ]

# Car brands (could also refine search by loading only certain models per brand)
#brand_list = ['audi', 'bmw', 'ford', 'mercedes', 'opel', 'skoda', 'toyota', 'volkswagen', 'volvo']
brand_list = [         'bmw',                                                'volkswagen']

# The autoscout24 website does not show more than 20 pages (and ~20 cars per page, so search is complete only if filters result in about 400 cars)
# To circumvent this, the search loops over sufficiently small price bins that there are never > 400 cars inside each; this guarantees we catch practically all cars
price_bin_edges = ['2500', '5000', '10000', '12500', '15000', '17500', '20000', '25000', '30000', '40000', '50000', '75000', '100000']

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

city_list = [ ['münchen'    , 'lat=48.13913&lon=11.58022'],
              ['berlin'     , 'lat=52.52343&lon=13.41144'] ]


