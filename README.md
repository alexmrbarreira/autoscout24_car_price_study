# autoscout24_prices_study

A study of what determines the price of used cars on sale in the german website [autoscout24.de](https://www.autoscout24.de/).

These python scripts execute two main tasks: 
1. scrape the autoscout24.de webpages to extract car data for different german cities, car brands and body types; it uses the python library BeautifulSoup.
2. train a series of machine learning models to predict car prices and study the relative importance of different car features to the final price.

This figure shows the outcome of one of the best models. It shows that:

1. the engine power is the most important feature in setting the car price, followed by the car year and the number of kilometers. 
2. Beyond that, the fuel type (diesel, petrol, eletric), type of transmission (auto vs. manual), car brand and chassis type have also a visible importance.
3. The warranty type, city, number of owners and seller type (handler vs. private) play a negligible role.

<img src="fig_store/fig_feature_importances_by_randomization_model_4_random_forest.png" width="600" height=auto class='center'/>
