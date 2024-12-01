# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 21:58:47 2024

@author: avina
"""

import pandas as pd
import numpy as np

df_manf = pd.read_csv('P:/My Documents/Books & Research/Analytics Vidya Blackbelt program/EDA/Manufacturing Dataset.csv')
df_manf.columns
df_manf.dtypes

#1 Dealing with IQR using outlier

num_cols = df_manf.select_dtypes(include = ['int', 'float'])
Q1 = num_cols.quantile(0.25)
Q3 = num_cols.quantile(0.75)
IQR = Q3 - Q1

outliers = ((num_cols < Q1 - 1.5 * IQR | (num_cols > (Q3 + 1.5 * IQR))))
outliers_summary = outliers.sum()
print(outliers_summary)

#2 Identify Missing Values Across Key Production Metrics

missing_values = df_manf.isnull().sum()
print(missing_values)
columns_to_fill = ['Defects', 'Maintenance Hours', 'Down time Hours','Rework Hours', 'DefectRate' ]

for col in columns_to_fill:
    df_manf[col].fillna(df_manf[col].mean(), inplace=True)

#3  Relationship Between Costs

import seaborn as sns
import matplotlib.pyplot as plt

# Scatter plot for material cost vs labor cost
sns.scatterplot(x='Material Cost Per Unit', y='Labour Cost Per Hour', data=df_manf)
plt.title('Material Cost vs Labor Cost')
plt.show()

# Calculate correlation coefficient
correlation = df_manf['Material Cost Per Unit'].corr(df_manf['Labour Cost Per Hour'])
print(f"Correlation Coefficient: {correlation}")

#4 Efficiency Across Shifts

# Group data by shifts
shift_efficiency = df_manf.groupby('Shift').agg(Mean_Prod_Time = ('Production Time Hours', 'mean'),
                                                Median_Prod_Time =('Production Time Hours','median'), 
                                                Mean_Energy_Usage = ('Energy Consumption kWh', 'mean'), 
                                                Median_Energy_Usage =('Energy Consumption kWh','median')
                                                ).reset_index()

print("Efficiency Metrics by Shift:")
print(shift_efficiency)


# Visualization
shift_efficiency.plot(kind='bar', figsize=(10, 5), title='Shift Efficiency Metrics')
plt.ylabel('Values')
plt.show()

# Statistical Tests (One-Way ANOVA)
# Test differences in ProductionTime

from scipy.stats import f_oneway

day_time = df_manf[df_manf['Shift'] == 'Day']['Production Time Hours']
swing_time = df_manf[df_manf['Shift'] == 'Swing']['Production Time Hours']
night_time = df_manf[df_manf['Shift'] == 'Night']['Production Time Hours']

f_stat_time, p_value_time = f_oneway(day_time, swing_time, night_time)
print(f"ANOVA Results for ProductionTime: F-statistic = {f_stat_time}, p-value = {p_value_time}")

# Test differences in EnergyUsed
day_energy = df_manf[df_manf['Shift'] == 'Day']['Energy Consumption kWh']
swing_energy = df_manf[df_manf['Shift'] == 'Swing']['Energy Consumption kWh']
night_energy = df_manf[df_manf['Shift'] == 'Night']['Energy Consumption kWh']

f_stat_energy, p_value_energy = f_oneway(day_energy, swing_energy, night_energy)
print(f"ANOVA Results for EnergyUsed: F-statistic = {f_stat_energy}, p-value = {p_value_energy}")

# Interpret results
if p_value_time < 0.05:
    print("Significant differences exist in ProductionTime across shifts.")
else:
    print("No significant differences found in ProductionTime across shifts.")

if p_value_energy < 0.05:
    print("Significant differences exist in EnergyUsed across shifts.")
else:
    print("No significant differences found in EnergyUsed across shifts.")

#5 Monthly Production Trends

# Extract month from date and group by month
df_manf['Product Type'].unique()
df_manf['Month'] = pd.to_datetime(df_manf['Date']).dt.strftime('%b')
monthly_trends = df_manf.groupby('Month')['Units Produced'].mean()
months_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
monthly_trends = monthly_trends.reindex(months_order)


# Plot monthly trends
monthly_trends.plot(kind='line', figsize=(10, 5), title='Monthly Production Trends')
plt.ylabel('Average Units Produced')
plt.show()

#  group by product and month
monthly_trends_pdt = df_manf.groupby(['Month', 'Product Type'])['Units Produced'].mean().unstack()
monthly_trends_pdt = monthly_trends_pdt.reindex(months_order)

# Plot monthly trends by Product type
monthly_trends_pdt.plot(kind='line', figsize=(12, 6), marker='o', title='Monthly Production Trends by Product Type')
plt.xlabel('Month')
plt.ylabel('Average Units Produced')
plt.xticks(rotation=45)
plt.legend(title='Product Type')
plt.grid(True)
plt.tight_layout()
plt.show()



#6 Variability in Production by Product Type

# Calculate standard deviation of production volume by product type
product_variability = df_manf.groupby('Product Type')['Production Volume Cubic Meters'].std()
product_variability_units = df_manf.groupby('Product Type')['Units Produced'].std()
print("Production Variability by Product Type:")
print(product_variability)
print(product_variability_units)

#7 The Role of Operator Count in Efficiency

# Group data by number of operators and calculate efficiency
df_manf['Units Produced per hr'] = df_manf['Units Produced'] / df_manf['Production Time Hours']
operator_efficiency = df_manf.groupby('Operator Count')['Units Produced per hr'].mean()

print("Operator Efficiency:")
print(operator_efficiency)

# Plot
operator_efficiency.plot(kind='bar', figsize=(10, 5), title='Operator Efficiency')
plt.ylabel('Units Produced Per Hour')
plt.show()

#8 Identifying the Machine with Most Defects

# Calculate defect rate per machine
df_manf['DefectRate'] = (df_manf['Defects'] / df_manf['Units Produced']) * 100
machine_defect_rate = df_manf.groupby('Machine ID')['DefectRate'].mean()

print("Defect Rates by Machine:")
print(machine_defect_rate)

#9 How Environment Affects Scrap Rate

# Correlation between temperature, humidity, and scrap rate
correlation_temp = df_manf['Average Temperature C'].corr(df_manf['Scrap Rate'])
correlation_humidity = df_manf['Average Humidity Percent'].corr(df_manf['Scrap Rate'])

print(f"Correlation with Temperature: {correlation_temp}")
print(f"Correlation with Humidity: {correlation_humidity}")

# Scatter plots
sns.scatterplot(x='Average Temperature C', y='Scrap Rate', data=df_manf)
plt.title('Temperature vs Scrap Rate')
plt.show()

sns.scatterplot(x='Average Humidity Percent', y='Scrap Rate', data=df_manf)
plt.title('Humidity vs Scrap Rate')
plt.show()

df_manf['Change in Temperature C'] = df_manf['Average Temperature C'].diff()
df_manf['Average Humidity change'] = df_manf['Average Humidity Percent'].diff()

sns.scatterplot(x='Change in Temperature C', y='Scrap Rate', data=df_manf)
plt.title('Change in Temperature vs Scrap Rate')
plt.show()

sns.scatterplot(x='Average Humidity change', y='Scrap Rate', data=df_manf)
plt.title('Change in Humidity vs Scrap Rate')
plt.show()

correlation_tempchange = df_manf['Change in Temperature C'].corr(df_manf['Scrap Rate'])
correlation_humiditychange = df_manf['Average Humidity change'].corr(df_manf['Scrap Rate'])
print(f"Correlation with change in Temperature: {correlation_tempchange}")
print(f"Correlation with change in Humidity: {correlation_humiditychange}")


# Maintenance hours vs Downtime hours

# Scatter plot
plt.figure(figsize=(8, 5))
plt.scatter(df_manf['Maintenance Hours'], df_manf['Down time Hours'], color='blue', alpha=0.7)
plt.title('Impact of Maintenance on Downtime')
plt.xlabel('Maintenance Hours')
plt.ylabel('Downtime Hours')
plt.grid(alpha=0.3)
plt.show()

# Correlation analysis
from scipy.stats import spearmanr


spearman_corr, spearman_p = spearmanr(df_manf['Maintenance Hours'], df_manf['Down time Hours'])
print(f"Spearman Correlation: {spearman_corr:.2f}, p-value: {spearman_p:.4f}")

if spearman_p < 0.05:
    print("The Spearman correlation is statistically significant.")
else:
    print("The Spearman correlation is not statistically significant.")
    

# Quality Control and Defect Rate

import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Independent variable (QualityChecksFailed) and dependent variable (DefectRate)
X = df_manf['Quality Checks Failed'].values.reshape(-1, 1)  # Reshape for regression
y = df_manf['DefectRate'].values

# Perform Linear Regression
model = LinearRegression()
model.fit(X, y)

# Predicted values
y_pred = model.predict(X)

# Add regression line to the scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(df_manf['Quality Checks Failed'], df_manf['DefectRate'], color='blue', label='Actual')
plt.plot(df_manf['Quality Checks Failed'], y_pred, color='red', label='Regression Line')
plt.title('Effect of Quality Checks Failed on Defect Rate')
plt.xlabel('Quality Checks Failed')
plt.ylabel('Defect Rate')
plt.legend()
plt.show()

# Print regression coefficients and R-squared
print("Regression Coefficients:")
print(f"Intercept: {model.intercept_}")
print(f"Slope: {model.coef_[0]}")
print(f"R-squared: {r2_score(y, y_pred)}")

# Using statsmodels for detailed summary
X_sm = sm.add_constant(X)  # Adds a constant term for the intercept
sm_model = sm.OLS(y, X_sm).fit()
print("\nStatsmodels Summary:")
print(sm_model.summary())