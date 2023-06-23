from parameters import *

# This script prepares the car data for training
#   It numerically encodes string variables and deals with missing values
#   It splits into training and validation sets

# Function that label encodes
def label_encode_variable(df, feature):
    le = preprocessing.LabelEncoder()
    le.fit(df[feature].values)
    df[feature] = le.transform(df[feature].values)
    print ('Label encoding of '+feature+': ' + str(le.classes_) + ' ----> ' + str(le.transform(le.classes_)))
    return le

# Function that applies data curation
def prepare_data(filename):
    df      = pd.read_csv(filename)
    df_size = df.shape[0]

    print ('')
    print ('The original data set has the following columns:')
    print (df.columns.tolist())

    nan_cols = df.columns[df.isnull().any()].tolist()
    print ('')
    print ('The followingy columns have NaN:')
    print (nan_cols)

    # Remove entries with nan in the price
    n0 = df.shape[0]
    df = df.dropna(subset=['Price[1000Eur]'])
    n1 = df.shape[0]
    print ('')
    print ('Removed', n0-n1, 'rows that had price=nan')

    # Keep only cars with < 300,000km (to remove severe outliers (e.g.~taxis))
    n0 = df.shape[0]
    df = df.loc[df['1000Km'] < 300.]
    n1 = df.shape[0]
    print ('')
    print ('Removed', n0-n1, 'rows with 1000Km > 300')

    # Replace nan in Owners with most common, and Warranty with 'Nein'
    most_common_owners = df['Owners'].mode()[0]
    df['Owners']       = df['Owners'].fillna(most_common_owners)
    df['Warranty']     = df['Warranty'].fillna('Nein')
    print ('')
    print ('In Owners, replaced nan with most common')
    
    # In Warranty: if it specifies months assume warranty (1), otherwise assume no warranty (0)
    #df['Warranty'].loc[df['Warranty'] == 'Ja']   = 'Nein'
    df['Warranty'].loc[df['Warranty'].str.contains('Monate', na=False)] = 'Ja-exp'
    print ('')
    print ('In Warranty, replaced month specification with "Ja-exp"')

    # In Gas, group all non diesel/benzin into Other (dominated by electric/hybrid)
    df['Gas'].loc[df['Gas'].str.contains('Elektro'       , na=False)] = 'Other'
    df['Gas'].loc[df['Gas'].str.contains('Elektro/Benzin', na=False)] = 'Other'
    df['Gas'].loc[df['Gas'].str.contains('Elektro/Diesel', na=False)] = 'Other'
    df['Gas'].loc[df['Gas'].str.contains('Erdgas'        , na=False)] = 'Other'
    df['Gas'].loc[df['Gas'].str.contains('Autogas'       , na=False)] = 'Other'
    df['Gas'].loc[df['Gas'].str.contains('Ethanol'       , na=False)] = 'Other'
    df['Gas'].loc[df['Gas'].str.contains('Wasserstoff'   , na=False)] = 'Other'
    df['Gas'].loc[df['Gas'].str.contains('Sonstige'      , na=False)] = 'Other'
    df['Gas'].loc[df['Gas'].str.contains('-'             , na=False)] = 'Other'
    print ('')
    print ('In Gas, grouped all non-diesel/benzin into Other (dominated by electric and hybrids)')

    # In Transmission, if km are specified assume it is elektro/hybrid and so automatik 
    df['Transmission'].loc[df['Transmission'].str.contains('km', na=False)]            = 'Automatik'
    df['Transmission'].loc[df['Transmission'].str.contains('-', na=False)]             = 'Automatik'
    df['Transmission'].loc[df['Transmission'].str.contains('Halbautomatik', na=False)] = 'Automatik'

    # Label encode categorial variables
    print ('')
    le_city         = label_encode_variable(df, 'City') 
    le_brand        = label_encode_variable(df, 'Brand') 
    le_body         = label_encode_variable(df, 'Body') 
    le_year         = label_encode_variable(df, 'Year') 
    le_gas          = label_encode_variable(df, 'Gas') 
    le_transmission = label_encode_variable(df, 'Transmission') 
    le_seller       = label_encode_variable(df, 'Seller') 
    le_owners       = label_encode_variable(df, 'Owners') 
    le_warranty     = label_encode_variable(df, 'Warranty') 

    # Drop URL column
    df              = df.drop(['URL'], axis = 1)

    print ('')
    print ('The final data set has the following columns:')
    print (df.columns.tolist())

    nan_cols = df.columns[df.isnull().any()].tolist()
    if (len(nan_cols) > 0):
        print ('')
        print ('Note the following columns still have NaN:')
        print (nan_cols)

    return df, le_city, le_brand, le_body, le_year, le_gas, le_transmission, le_seller, le_owners, le_warranty

print ('')
print ('Preparing data ...')
df_prepared, le_city, le_brand, le_body, le_year, le_gas, le_transmission, le_seller, le_owners, le_warranty = prepare_data('data_store/data_cars_autoscout24.csv')
#df_prepared, le_city, le_brand, le_body, le_year, le_gas, le_transmission, le_seller, le_owners, le_warranty = prepare_data('data_store/data_cars_autoscout24_munich_berlin_allbrands_btypes1456.csv')

# Save encoders for later use in transformations
pickle.dump(le_city        , open('encoder_store/le_city.pkl'        , 'wb'))
pickle.dump(le_brand       , open('encoder_store/le_brand.pkl'       , 'wb'))
pickle.dump(le_body        , open('encoder_store/le_body.pkl'        , 'wb'))
pickle.dump(le_year        , open('encoder_store/le_year.pkl'        , 'wb'))
pickle.dump(le_gas         , open('encoder_store/le_gas.pkl'         , 'wb'))
pickle.dump(le_transmission, open('encoder_store/le_transmission.pkl', 'wb'))
pickle.dump(le_seller      , open('encoder_store/le_seller.pkl'      , 'wb'))
pickle.dump(le_owners      , open('encoder_store/le_owners.pkl'      , 'wb'))
pickle.dump(le_warranty    , open('encoder_store/le_warranty.pkl'    , 'wb'))

# =======================================================
# Split into training/validation
# =======================================================

training_validation_split = 0.8

select   = np.random.rand(len(df_prepared)) < training_validation_split
df_train = df_prepared[select]
df_valid = df_prepared[~select]

print ('')
print ('Total data size:')
print (df_prepared.shape[0])
print ('Training data size:')
print (df_train.shape[0])
print ('Validation data size:')
print (df_valid.shape[0])

df_train.to_csv('data_store/data_prepared_train.csv', index = False)
df_valid.to_csv('data_store/data_prepared_valid.csv', index = False)

