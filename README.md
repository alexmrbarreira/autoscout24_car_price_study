# autoscout24_prices_study

A study of what determines the price of used cars on sale in the german website [autoscout24.de](https://www.autoscout24.de/).

These python scripts execute two main tasks: 
1. scrape the autoscout24.de webpages to extract car data for different german cities, car brands and body types.
2. train machine learning models to predict car prices and study the relative importance of different car features to the final price.

This figure shows the outcome of one of the best models:

1. the engine power is the most important feature in setting the car price, followed by the car year and the number of kilometers. 
2. Beyond that, the fuel type (diesel, petrol, eletric), type of transmission (auto vs. manual), car brand and chassis type have also a visible importance.
3. The warranty type, city, number of owners and seller type (handler vs. private) play a negligible role.

<img src="fig_store/fig_feature_importances_by_randomization_model_4_random_forest.png" width="600" height=auto/>

## Table of contents
- [Dependencies](#dependencies)
- [Code overview](#code-overview)
- [The car data](#the-car-data)
- [Machine learning model performance](#machine-learning-model-performance)

## Dependencies

- numpy, scipy and matplotlib
- pandas
- scikit-learn

## Code overview

To run the whole pipeline, execute as follows:

*python scrape_autoscout24_de.py; python prepare_training_data.py; python plot_data_stats.py; python train_regression_models.py; python plot_feature_importance.py*

#### parameters.py
This file defines the car search parameters. Edit it to choose which brands, chassis types and cities to browse on autoscout24.de.

Other global functions, parameters and library imports are also specified here. This file is called by all other files.

#### scrape_autoscout24_de.py
This file does the autoscout24.de scraping. The main search loop is:

```ruby
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
```

The function get_cars_URL() fir collects all desired car URL addresses, which the function save_car_data() then scrapes to extract the car properties. The car properties are:

| price | city | brand | body | km  | power | year | fuel | transmission | seller | owners | warranty type |
| :---: | :--: | :---: | :--: | :-: | :---: | :--: | :--: | :---------:  | :----: | :----: | :-----------: |

The data is saved in the file data_store/data_store/data_cars_autoscout24.csv. The folder data_store/ contains already data from some searches. To extract all of the data for a single city, 9 car brands and 5 chassis types it takes about 4-5h (depending on the internet speed and CPU).




