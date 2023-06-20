from parameters import *

# ================================================================ 
# Load data 
# ================================================================ 

df_train, df_valid, train_features, valid_features, train_labels, valid_labels, N_train, N_valid, N_featu = get_train_valid_data()

df_total = pd.concat([df_train, df_valid])
cols     = df_total.columns.tolist()

le_city, le_brand, le_body, le_year, le_gas, le_transmission, le_seller, le_owners, le_warranty = get_encoders()
list_of_le = [le_city, le_brand, le_body, None, None, None, le_year, le_gas, le_transmission, le_seller, le_owners, le_warranty]

#df = pd.read_csv('data_store/data_cars_autoscout24.csv')
#a = df['URL'].loc[df['Owners'] > 4]
#for i in a:
#    print (i)

# ================================================================ 
# Plot # of cars as a function of features
# ================================================================ 

fig0 = plt.figure(0, figsize = (17., 10.))
fig0.subplots_adjust(left=0.06, right=0.99, bottom=0.09, top = 0.98, hspace = 0.4, wspace = 0.2)

def plot_bincount(df, feature, le):
    # Add counts
    xx_names = le.classes_
    xx_encod = le.transform(xx_names)
    xx       = range(len(xx_names))
    counts   = np.bincount(df[feature].values)
    plt.bar(xx, counts, color = 'g', alpha = alpha_c, label = '\# of cars')
    # Add mean car price trend in feature valurs (normalized by max count)
    mean_price = np.zeros(len(counts))
    for i in range(len(counts)):
        mean_price[i] = df['Price[1000Eur]'].loc[df[feature]==xx_encod[i]].mean()
    mean_price = mean_price * max(counts) / max(mean_price) * 0.9
    plt.plot(xx, mean_price, linewidth = 2., linestyle = 'dashed', c = 'darkorange', label = 'Price trend')
    # Cosmetics
    plt.xticks(range(len(xx_names)), xx_names, rotation = 30.)
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
    plt.bar(bin_means, counts, color = 'g', alpha = alpha_c, width = (9./10)*width, label = '\# of cars')
    # Add mean car price trend in feature bin (normalized by max count)
    mean_price = np.zeros(nbins)
    for i in range(nbins):
        mean_price[i] = df['Price[1000Eur]'].loc[ (df[feature] >= bin_edges[i]) & (df[feature] < bin_edges[i+1]) ].mean()
    mean_price = mean_price * max(counts) / max(mean_price) * 0.9
    plt.plot(bin_means, mean_price, linewidth = 2., linestyle = 'dashed', c = 'darkorange', label = 'Price trend')
    # Cosmetics
    plt.xlabel(feature, fontsize = label_font)
    plt.tick_params(length=tick_major, width=tickwidth , bottom=True, top=True, left=True, right=True, direction = 'in', which = 'major', pad = tickpad, labelsize = ticksize)
    return 0

for i in range(len(cols)):
    fig0.add_subplot(3,4,i+1)
    # Add data
    if ( (i==3) or (i==4) or (i==5) ):
        plot_hist(df_total, cols[i])
    else:
        plot_bincount(df_total, cols[i], list_of_le[i])
    # Cosmetics
    if ( (i==0) or (i==4) or (i==8) ):
        plt.ylabel('\# of cars', fontsize = label_font)
    if ( (i==10) or (i==11) ):
        plt.xlabel(cols[i], fontsize = label_font)
    if (i == 4):
        params = {'legend.fontsize': legend_font-4}; plt.rcParams.update(params); plt.legend(loc = 'upper right', ncol = 1)

fig0.savefig('fig_store/fig_data_stats_counts.png')

# ================================================================ 
# Plot correlation matrix  
# ================================================================ 

fig1 = plt.figure(1, figsize = (12., 12.))
fig1.subplots_adjust(left=0.10, right=0.98, bottom=0.08, top=0.94)

corr_matrix = np.corrcoef(np.transpose(df_total.values))

fig1.add_subplot(1,1,1)
# Add correlation matrix data
plt.imshow(corr_matrix, cmap = 'seismic', vmin=-1, vmax=1)
# Add number to plot
for i in range(len(corr_matrix[:,0])):
    for j in range(i):
        corr_now = round(corr_matrix[i,j], 2)
        if (corr_now >= 0):
            plt.annotate(r'$'+str(corr_now)+'$', xy = (j-0.25, i+0.1), xycoords = 'data', c = 'k', fontsize = text_font-6)
        else:
            plt.annotate(r'$'+str(corr_now)+'$', xy = (j-0.40, i+0.1), xycoords = 'data', c = 'k', fontsize = text_font-6)
# Cosmetics
plt.title('Correlation matrix', fontsize = title_font+4)
plt.xticks(range(len(cols)), cols, fontsize = ticksize, rotation = 45.)
plt.yticks(range(len(cols)), cols, fontsize = ticksize, rotation = 45.)
cb = plt.colorbar()
cb.set_label(r'$r_{ij} = {\rm Cov}_{ij}/\sqrt{{\rm Cov}_{ii}{\rm Cov}_{jj}}$', fontsize = label_font)
cb.ax.tick_params(labelsize = ticksize)

fig1.savefig('fig_store/fig_data_stats_correlation.png')

#plt.show()
