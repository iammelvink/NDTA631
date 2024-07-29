# %% [markdown]
# ## Py-imp is the section of the repo that is implemented using Python
# - This notebook is the practical_01 data analysis and visualisation section

# %%
# Warning options (before all imports)
import warnings
warnings.filterwarnings('ignore')
# %xmode Verbose # simplified traceback when an exception occurs
%xmode Plain

# %% [markdown]
# ### Install needed packages

# %%
# Install a conda package in the current Jupyter kernel
import sys
# !conda install -c conda-forge --yes --prefix {sys.prefix} <pkg>

# Install a pip package in the current Jupyter kernel
# import sys
# !{sys.executable} -m pip install <pkg>

# %%
# # upgrade pip
# !{sys.executable} -m pip install --upgrade pip

# %%
# !{sys.executable} -m pip install pandas==1.5.3
# !{sys.executable} -m pip install numpy
# !{sys.executable} -m pip install matplotlib
# !{sys.executable} -m pip install seaborn
# !{sys.executable} -m pip install watermark
# !{sys.executable} -m pip install squarify

# %% [markdown]
# ### Importing libraries

# %%
import os, random

import numpy as np
import squarify
from math import pi

# EDA
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# To display charts in Jupyter
%matplotlib inline

# For reproducibility
rng = 777
os.environ['PYTHONHASHSEED'] = str(rng)
random.seed(rng)
np.random.seed(rng)

%load_ext autoreload
%autoreload 2

# %%
%load_ext watermark
# %watermark
%watermark --iversions

# %% [markdown]
# <a id="import-and-clean-data"></a>
# ## 1.   Import and Clean Data

# %%
dataset = sns.load_dataset("diamonds")

# %% [markdown]
# ### Diamond Dataset Variables
# 
# - **carat**: Weight of the diamond
# - **cut**: Quality of the diamond's cut (shape and symmetry)
# - **color**: Colour of the diamond
# - **clarity**: How clear and free of imperfections the diamond is
# - **depth**: Height of diamond, as percentage of its width
# - **table**: Width of diamond's top surface, as percentage
# - **price**: How much the diamond costs
# - **x**: Length of the diamond in millimeters
# - **y**: Width of the diamond in millimeters
# - **z**: Depth of the diamond in millimeters

# %% [markdown]
# <a id="data-analysis"></a>
# ## 2.   Data Analysis

# %%
## Display the first 5 rows of the dataset
dataset.head(5)

# %%
## Calculate summary statistics for numeric columns
summary_stats = dataset.describe()
summary_stats

# %%
## Find the mean price of diamonds
mean_price = dataset['price'].mean()
mean_price

# %%
## Group by 'cut' and find the average price for each cut
avg_price_by_cut = dataset.groupby('cut')['price'].mean()
avg_price_by_cut

# %%
## Find the most common 'colour' in the dataset
most_common_color = dataset['color'].mode()[0]
most_common_color

# %%
## Find the correlation between 'carat' and 'price'
correlation = dataset['carat'].corr(dataset['price'])
correlation

# %%
## Calculate the median depth of diamonds
median_depth = dataset['depth'].median()
median_depth

# %% [markdown]
# <a id="data-visualisation"></a>
# ## 3.   Data Visualisation

# %%
## Basic Charts
## Plot a bar chart of the average price by cut
avg_price_by_cut.plot(kind='bar')
plt.title('Average Price by Cut')
plt.xlabel('Cut')
plt.ylabel('Average Price')
plt.show()

# %%
## Create a line chart of the average price over different carat values
avg_price_by_carat = dataset.groupby('carat')['price'].mean().reset_index()
plt.plot(avg_price_by_carat['carat'], avg_price_by_carat['price'])
plt.title('Average Price over Carat Values')
plt.xlabel('Carat')
plt.ylabel('Average Price')
plt.show()

# %%
## Plot a donut chart of the count of diamonds by cut
cut_counts = dataset['cut'].value_counts()
plt.pie(cut_counts, labels=cut_counts.index, startangle=90, wedgeprops={'width': 0.3})
plt.title('Count of Diamonds by Cut')
plt.show()

# %%
## Statistical Visualisations
## Create a scatter plot of 'carat' vs 'price'
plt.scatter(dataset['carat'], dataset['price'])
plt.title('Scatter Plot of Carat vs Price')
plt.xlabel('Carat')
plt.ylabel('Price')
plt.show()

# %%
## Plot a histogram of diamond prices
plt.hist(dataset['price'], bins=50)
plt.title('Histogram of Diamond Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

# %%
## Plot a box plot for 'price' by 'cut'
sns.boxplot(x='cut', y='price', data=dataset)
plt.title('Box Plot of Price by Cut')
plt.xlabel('Cut')
plt.ylabel('Price')
plt.show()

# %%
## Complex Data Representations
## Create a heatmap for the correlation matrix of the dataset
corr_matrix = dataset.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix Heatmap')
plt.show()

# %%
## Plot a treemap for the count of diamonds by colour
color_counts = dataset['color'].value_counts()
squarify.plot(sizes=color_counts, label=color_counts.index, alpha=.8)
plt.axis('off')
plt.title('Treemap of Diamonds by Colour')
plt.show()

# %%
## Specialised Charts
## Create a radar chart for the mean values of numerical columns

# Make a list of number columns
categories = list(dataset.select_dtypes(include=[np.number]).columns)

# Count how many number columns there are
N = len(categories)

# Calculate average of each number column
values = dataset[categories].mean().tolist()

# Add the first value to the end
values += values[:1]

# Create points around a circle
angles = [n / float(N) * 2 * pi for n in range(N)]

# Add the first angle to the end
angles += angles[:1]

# Draw lines between the points
plt.polar(angles, values)

# Color inside the lines
plt.fill(angles, values, alpha=0.3)

# Label each point with its category name
plt.xticks(angles[:-1], categories)

# Add a title to the chart
plt.title('Radar Chart of Mean Values')

# Show the chart
plt.show()

# %%
## Plot a stacked bar chart of the count of diamonds by cut and colour
cut_color_counts = dataset.groupby(['cut', 'color']).size().unstack()
cut_color_counts.plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Count of Diamonds by Cut and Colour')
plt.xlabel('Cut')
plt.ylabel('Count')
plt.show()


