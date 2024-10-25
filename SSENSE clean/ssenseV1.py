'''
Eric Tech
COMSC225
Final Project.SSENSE
---------------------
This dataset includes product listings from SSENSE.com (2024).
It includes the fashion brand, description (highlighting key features), price in USD and type (male or female).
This dataset could be used for trend analysis in luxury fashion, gender based market insight, or brand and price segmentation.
Question: How do woman and mens clothing prices compare?
---------------------
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Save CSV into Dataframe
data = pd.read_csv('ssense.csv')
data['price_usd'] = pd.to_numeric(data['price_usd'], errors='coerce')

# Cleaning Data
data.drop_duplicates(inplace=True)
data = data.dropna(how='any')

# Remove outliers w/(IQR) method
Q1 = data['price_usd'].quantile(0.25)
Q3 = data['price_usd'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
data_filtered = data[(data['price_usd'] >= lower_bound) & (data['price_usd'] <= upper_bound)]

# Print Stats of all data
print("Price Mean (after outlier removal): ", data_filtered['price_usd'].mean())
print("Price Minimum (after outlier removal): ", data_filtered['price_usd'].min())
print("Price Maximum (after outlier removal): ", data_filtered['price_usd'].max())
print("Price Standard Deviation (after outlier removal): ", data_filtered['price_usd'].std())

# Grouped data for brand prices of all data
print("Grouped Data (after outlier removal): \n", data_filtered.groupby('brand')['price_usd'].describe())

# Regression Scatter Plot for all data
x = np.arange(len(data_filtered))
y = data_filtered['price_usd']
plt.title('Cost of Pieces on SSENSE (After Outlier Removal)')
plt.xlabel('Brands')
plt.ylabel('Price in USD')
plt.grid(True)
slope, intercept, r, p, std_err = stats.linregress(x, y)
regmodel = slope * x + intercept
plt.scatter(x, y)  # scatter plot
plt.scatter(x, regmodel, color='red')   # regression line
plt.show()

# Filter data for men's clothing ONLY
data_men = data_filtered[data_filtered['type'] == 'mens']
data_men_sampled = data_men.sample(n=8000, random_state=42)
price_men_sampled = data_men_sampled['price_usd']

# Filter data for women's clothing ONLY
data_women = data_filtered[data_filtered['type'] == 'womens']
price_women = data_women['price_usd']

# Regression Scatter Plot for men prices
x_men_sampled = np.arange(len(data_men_sampled))
y_men_sampled = price_men_sampled
plt.scatter(x_men_sampled, y_men_sampled, label='Men Prices')
plt.title('Cost of Men Pieces on SSENSE')
plt.xlabel('Index')
plt.ylabel('Price in USD')
plt.grid(True)
slope_men_sampled, intercept_men_sampled, r_men_sampled, p_men_sampled, std_err_men_sampled = stats.linregress(x_men_sampled, y_men_sampled)
regmodel_men_sampled = slope_men_sampled * x_men_sampled + intercept_men_sampled
plt.plot(x_men_sampled, regmodel_men_sampled, color='red')  # Regression line
plt.show()

# Regression Scatter Plot for women prices
x_women = np.arange(len(data_women))
y_women = data_women['price_usd']
plt.scatter(x_women, y_women, label='Women Prices')
plt.title('Cost of Women Pieces on SSENSE')
plt.xlabel('Index')
plt.ylabel('Price in USD')
plt.grid(True)
slope_women, intercept_women, r_women, p_women, std_err_women = stats.linregress(x_women, y_women)
regmodel_women = slope_women * x_women + intercept_women
plt.plot(x_women, regmodel_women, color='red')  # Regression line
plt.show()

# Mean/STD of men & women
print("\nWomen's Statistics: \nMean Price of Women's Clothing: ", int(price_women.mean()))
print("Price Standard Deviation: ", price_women.std())

print("\nMen's Statistics: \nMean Price of Men's Clothing: ", int(price_men_sampled.mean()))
print("Price Standard Deviation: ", price_men_sampled.std())