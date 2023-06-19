from parameters import *

# ================================================================ 
# Load data 
# ================================================================ 

df_train, df_valid, train_features, valid_features, train_labels, valid_labels, N_train, N_valid, N_featu = get_train_valid_data()

df_total = pd.concat([df_train, df_valid])
cols     = df_total.columns.tolist()

le_city, le_brand, le_body, le_gas, le_transmission, le_seller, le_warranty = get_encoders()
list_of_le = [le_city, le_brand, le_body, None, None, None, None, le_gas, le_transmission, le_seller, None, le_warranty]

#df = pd.read_csv('data_store/data_cars_autoscout24.csv')
#a = df['URL'].loc[df['1000Km'] > 500]
#for i in a:
#    print (i)

# ================================================================ 
# Plot # of cars as a function of features
# ================================================================ 

fig0 = plt.figure(0, figsize = (17., 10.))
fig0.subplots_adjust(left=0.06, right=0.99, bottom=0.09, top = 0.98, hspace = 0.4, wspace = 0.2)

def plot_bincount(df, feature, le):
    xx_names = le.classes_
    xx       = range(len(xx_names))
    counts   = np.bincount(df[feature].values)
    plt.bar(xx, counts, color = 'g', alpha = alpha_c)
    plt.xticks(range(len(xx_names)), xx_names, rotation = 20.)
    plt.tick_params(length=tick_major, width=tickwidth , bottom=True, top=True, left=True, right=True, direction = 'in', which = 'major', pad = tickpad, labelsize = ticksize)
    return 0

def plot_hist(df, feature):
    # Add counts
    values     = df[feature].values
    nbins      = 20
    bin_edges  = np.linspace(min(values), max(values), nbins+1)
    bin_means  = (bin_edges[1::] + bin_edges[0:-1])/2.
    width      = bin_means[1] - bin_means[0]
    counts     = np.histogram(values, bins = bin_edges)[0]
    plt.bar(bin_means, counts, color = 'g', alpha = alpha_c, width = (9./10)*width)
    # Add mean car price trend in feature bin (normalized by max count)
    mean_price = np.zeros(nbins)
    for i in range(nbins):
        mean_price[i] = df['Price[1000Eur]'].loc[ (df[feature] >= bin_edges[i]) & (df[feature] < bin_edges[i+1]) ].mean()
    mean_price = mean_price * max(counts) / max(mean_price)
    plt.plot(bin_means, mean_price, linewidth = 2., linestyle = 'dashed', c = 'darkorange')
    # Cosmetics
    plt.xlabel(feature, fontsize = label_font)
    plt.tick_params(length=tick_major, width=tickwidth , bottom=True, top=True, left=True, right=True, direction = 'in', which = 'major', pad = tickpad, labelsize = ticksize)
    return 0

for i in range(len(cols)):
    fig0.add_subplot(3,4,i+1)
    if ( (i==3) or (i==4) or (i==5) or (i==6) or (i==10) ):
        plot_hist(df_total, cols[i])
    else:
        plot_bincount(df_total, cols[i], list_of_le[i])
    if ( (i==0) or (i==4) or (i==8) ):
        plt.ylabel(r'\# of cars', fontsize = label_font)


plt.show()
