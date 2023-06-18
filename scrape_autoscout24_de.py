
from parameters_de import *

# This script scrapes the autoscout24.de webpages for car data
#   It first selects car URLs that match the desired search criteria
#   Then it loops over them to scrape the desired car data
#   Note this is very specific to the autoscout24.de html -- for other car websites, this needs to be modified accordingly.

# ======================================================= 
# Loop over autoscout24 search pages to collect URLs of individual car offers
# ======================================================= 
cars_URL = []

# Loop over the cities
for city in city_list:
    print ('Doing city', city[0], 'out of', [a[0] for a in city_list])
    
    # Loop over the brands
    for brand in brand_list: # loop over brands
        print ('    Doing brand', brand, 'out of', brand_list)

        # Loop over the body types
        for body in body_types: # loop over body types
            print ('        Doing body type', body, 'out of', body_types)

            # Loop over the price bins
            for iprice in range(len(price_bin_edges)-1):
                print ('            On price bin', price_bin_edges[iprice], 'to', price_bin_edges[iprice+1])

                # Loop over the pages
                k_counter = 0
                for page in range(1, max_page+1): # loop over pages

                    # Call to main URL from which to collect car URLs (save all links - a tags - which are scraped later)
                    try:
                        url = 'https://www.autoscout24.de/lst/'+brand+'/'+city[0]+'?atype=C&cy=D&desc=0&fregfrom='+min_year+'&'+city[1]+'&ocs_listing=include&page='+str(page)+'&pricefrom='+price_bin_edges[iprice]+'&priceto='+price_bin_edges[iprice+1]+'&search_id=z3rlqsigeg&sort=standard&source=listpage_pagination&ustate=N%2CU&zipr='+search_radius
                        only_a_tags = SoupStrainer("a")
                        soup = BeautifulSoup(requests.get(url).text,'lxml', parse_only=only_a_tags)
                    except Exception as e:
                        print("Request to URL failed: " + str(e) +" "*50, end="\r")
                        pass

                    # From the soup objects, keep only URLs with "angebote" ("offers")
                    for link in soup.find_all("a"):
                        target_url = str(link.get("href"))
                        if (r"/angebote/" in target_url) and (r"/leasing/" not in target_url): # include normal offers (exc. smyle offers -- autoscout online buying system)
                            if( (brand not in target_url) ):
                                print ('                The brand string does not appear in the car URLs; could signal that the search failed. Check input parameters.')
                                print ('                brand, URL = ', brand, target_url)
                            cars_URL.append('https://www.autoscout24.de/' + target_url)
                            k_counter += 1
                if(k_counter > 400):
                    print ('')
                    print ('                This price range has', k_counter, 'cars')
                    print ('                As a rule of thumb autoscout shows 20 pages with 20 cars per page ~ 400 cars; some cars might have been missed.')
                    print ('')
            print ('        Number of cars selected by now = ', len(cars_URL))

# Remove duplicates
cars_URL = list(dict.fromkeys(cars_URL))

print ('Total number of car URLs', len(cars_URL))

# ======================================================= 
# Loop over car URLs and parse desired information
# ======================================================= 
cars_data = []
counter = 1
for url in cars_URL:
    dict_now = {}
    if (np.mod(counter, int(len(cars_URL)/10))==0):
        print ('Done collecting data for', counter, 'cars')

    try:
        car         = BeautifulSoup(requests.get('https://www.autoscout24.de' + url).text, 'lxml')
        price_parse = car.find_all("span", attrs={"class":"PriceInfo_price__JPzpT"})
        other_parse = car.find_all("div", attrs={"class":"VehicleOverview_itemText__V1yKT"})

        # Get the attributes: {price, km, transmission, year, gas, power} (playing string tricks here to get desired data)
        price = price_parse[0].text.split()[1].split(',')[0]
        km    = other_parse[0].text.split()[0]
        trans = other_parse[1].text
        year  = other_parse[2].text.split('/')[1]
        gas   = other_parse[3].text.split()[0]
        power = other_parse[4].text.split()[2].split('(')[1]

        cars_data.append([price, km, power, year, gas, trans, url])
        counter += 1

    except Exception as e:
        print ('Loading data failed for car URL', url, ':' + str(e) +" "*50, end="\r")
        counter += 1

# Save to file
df = pd.DataFrame(cars_data, columns=['Price', 'Km', 'Power', 'Year', 'Gas', 'Transmission', 'URL'])
df.to_csv(data_filename)

