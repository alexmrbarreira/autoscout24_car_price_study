from parameters import *

# This script scrapes the autoscout24.de webpages for car data
#   It first selects car URLs that match the desired search criteria
#   Then it loops over them to scrape the desired car data
#   Note this is very specific to the autoscout24.de html -- for other car websites, this needs to be modified accordingly.

# ======================================================================
# Function that loops over autoscout24 search pages to collect carl URLs 
# ======================================================================
def get_cars_URL(city, brand, body):
    cars_URL = []

    # Loop over the price bins
    for iprice in range(len(price_bin_edges)-1):
        # Loop over the pages
        k_counter = 0
        for page in range(1, max_page+1): # loop over pages

            print ('\r                 On price bin '+price_bin_edges[iprice]+' to '+price_bin_edges[iprice+1]+', page ' + str(page), end = '', flush = True)

            # Call to main URL from which to collect car URLs (save all links - a tags - which are scraped later)
            try:
                url = 'https://www.autoscout24.de/lst/'+brand+'/'+city[0]+'?atype=C&body='+body+'&cy=D&desc=0&fregfrom='+min_year+'&'+city[1]+'&ocs_listing=include&page='+str(page)+'&pricefrom='+price_bin_edges[iprice]+'&priceto='+price_bin_edges[iprice+1]+'&search_id=z3rlqsigeg&sort=standard&source=listpage_pagination&ustate=N%2CU&zipr='+search_radius
                only_a_tags = SoupStrainer("a")
                soup = BeautifulSoup(requests.get(url).text,'lxml', parse_only=only_a_tags)
            except Exception as e:
                print("Request to URL failed: " + str(e) +" "*50, end="\r")
                pass

            # From the soup objects, keep only URLs with "angebote" ("offers")
            for link in soup.find_all("a"):
                target_url = str(link.get("href"))
                if (r"/angebote/" in target_url) and (r"/leasing/" not in target_url): # include normal offers (exc. smyle offers -- autoscout online buying system)
                    cars_URL.append('https://www.autoscout24.de' + target_url)
                    k_counter += 1

    if(k_counter > 400):
        print ('')
        print ('                This price range has', k_counter, 'cars')
        print ('                As a rule of thumb autoscout shows 20 pages with 20 cars per page ~ 400 cars; some cars might have been missed.')
        print ('')

    # Remove any duplicates
    cars_URL = list(dict.fromkeys(cars_URL))
    return cars_URL

# ======================================================================
# Function that scrapes car URLs and saves data to file
# ======================================================================
def save_car_data(filename, cars_URL, city, brand, body):
    cars_data = []
    fail_counter = 0
    k = 0.
    for url in cars_URL:
        dict_now = {}

        fraction_done = k/len(cars_URL) * 100.
        print ('\r                 Dealt with '+str(int(round(fraction_done, 0)))+'% of the URLs', end='', flush = True)

        try:
            car         = BeautifulSoup(requests.get(url).text, 'lxml')
            # Get car price (if condition keeps only cars with single price offer, ie., without monthly option only)
            price_parse = car.find_all("span", attrs={"class":"PriceInfo_price__JPzpT"})
            if('mtl' not in price_parse[0].text):
                price       = price_parse[0].text.split()[1].split(',')[0]
            else:
                price = float('nan')
            # Get main car features
            main_parse = car.find_all("div", attrs={"class":"VehicleOverview_itemText__V1yKT"})
            km         = main_parse[0].text.split()[0]
            if('.' in km): # write in 1000Km units
                km = float(km)
            else:
                km = float(km)/1000.
            trans      = main_parse[1].text
            year       = main_parse[2].text.split('/')[1]
            gas        = main_parse[3].text.split()[0]
            power      = main_parse[4].text.split()[2].split('(')[1]
            seller     = main_parse[5].text
            # Get number of owners 
            n_owners      = float('nan')
            history_parse = car.find_all("div", attrs={"class":"DetailsSection_container__kJAVE DetailsSection_breakElement__ODImO", "data-cy":"listing-history-section"})
            history_attrs = history_parse[0].find_all("dt")
            history_value = history_parse[0].find_all("dd")
            for i in range(len(history_attrs)):
                if(history_attrs[i].text == 'Fahrzeughalter'):
                    n_owners = history_value[i].text
            # Get warranty
            warranty     = 'nan'
            basics_parse = car.find_all("div", attrs={"class":"DetailsSection_container__kJAVE DetailsSection_breakElement__ODImO", "data-cy":"basic-details-section"})
            basics_attrs = basics_parse[0].find_all("dt")
            basics_value = basics_parse[0].find_all("dd")
            for i in range(len(basics_attrs)):
                if(basics_attrs[i].text == 'Garantie'):
                    warranty = basics_value[i].text

            #print ([city[0], brand, body_names[int(body)-1], price, km, power, year, gas, trans, seller, n_owners, warranty, url])

            cars_data.append([city[0], brand, body_names[int(body)-1], price, km, power, year, gas, trans, seller, n_owners, warranty, url])

        except Exception as e:
            fail_counter += 1
            #print ('')
            #print ('Loading data failed for car URL', url, ':' + str(e) +" "*50, end="\r")
            #print ('')

        k += 1

    # Save to file
    print ('. Failed to open', fail_counter, 'URLs')
    if (len(cars_data) > 0):
        df = pd.DataFrame(cars_data, columns=['City', 'Brand', 'Body', 'Price[1000Eur]', '1000Km', 'Power[HP]', 'Year', 'Gas', 'Transmission', 'Seller', 'Owners', 'Warranty', 'URL'])
        if os.path.exists(filename):
            df.to_csv(filename, mode = 'a', index = False, header = False)
        else:
            df.to_csv(filename, mode = 'w', index = False, header = True)

    return cars_data

# ======================================================================
# Execute the car data collection 
# ======================================================================

filename = 'data_store/data_cars_autoscout24.csv'

if os.path.exists(filename):
    print ('')
    print ('=====================================================================================================================')
    print ('*********************************************************************************************************************')
    print ('')
    print ('BEWARE! Note that the file', filename, 'already exists; this will append data to it potentially creating duplicates!')
    print ('')
    print ('*********************************************************************************************************************')
    print ('=====================================================================================================================')
    print ('')

car_counter = 0
# Loop over the cities
for city in city_list:
    print ('Doing city', city[0], 'out of', [a[0] for a in city_list])
    
    # Loop over the brands
    for brand in brand_list: # loop over brands
        print ('    Doing brand', brand, 'out of', brand_list)

        # Loop over the body types
        for body in body_types: # loop over body types
            print ('        Doing body type', body, 'out of', body_types)

            # Get car URLs
            print ('            Getting car URLs ... ')
            cars_URL = get_cars_URL(city, brand, body)
            print ('            ... done! Number of URLs:', len(cars_URL))

            # Get and save car data from URLs
            print ('            Getting car data ... ')
            cars_data = save_car_data(filename, cars_URL, city, brand, body)
            print ('            ... done! Appended to', filename)
            car_counter += len(cars_data)
            print ('            Total number of cars added:', car_counter)


