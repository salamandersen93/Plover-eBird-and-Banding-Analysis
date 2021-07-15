#!/usr/bin/env python
# coding: utf-8

# <b>eBird Data (Piping Plover) Data Cleaning</b>:
# 
# Data was obtained via https://ebird.org/data/download. A download request was created for Piping Plover data from 1960 - 2021.
# 
# This dataset will be incorporated into the final product to accompany the curated bird banding dataset from https://www.sciencebase.gov/catalog/item/60914db3d34e791692e13a22, also included in this project.
# 
# The eBird dataset contains reported sightings by citizen scientists reported on the eBird website. Reported data includes observation counts, observation date, breeding behavior, location, locality, time of observation, protocol type, and a variety of free text comments regarding the trip and species. The dataset provides valuable information around piping plover migration, breeding, and population status. Each entry has been vetted by subject matter experts to ensure accuracy and reliability, all non-vetted data was omitted from this dataset via the download portal. Observers occasionally report band information in the species comments field - this information can be mined to supplement the banding dataset referenced above.
# 
# The purpose of the code below is to apply cleaning procedures onthe dataset including: standardizing casing across strings + removing whitespace and newlines in select columns, correcting typos, standardizing dates, removing invalid data, and addressing missing data with interpolation or other methods.

# In[1]:


get_ipython().system(' pip install plotly.express --upgrade')


# In[2]:


import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandasql as ps


# In[3]:


# get raw data -- note common forms of N/A are being addressed while reading into the dataframe
missing_values = ["n/a", "na", "N/A", "NA" ,"--", "-"]
raw_data = pd.read_csv("Input_Files/ebird_pipplo_dataset.csv", na_values = missing_values)
pd.set_option('display.max_columns', None) # lots of columns, need to set to display max
raw_data.head()


# <b>Getting an understanding of where missing data exists and dropping unnecessary columns</b>:

# In[4]:


# separating by data type as missing data will need to be addressed in separate manners
object_data = raw_data.columns[raw_data.dtypes == 'object']
numeric_data= raw_data.columns[raw_data.dtypes != 'object']

print(object_data)
print(numeric_data)


# In[5]:


# getting percentages of rows with missing data in each column
raw_data[object_data].isnull().sum().sort_values(ascending=False)/len(raw_data)


# In[6]:


raw_data[numeric_data].isnull().sum().sort_values(ascending=False)/len(raw_data) # perentage of dataframe with missing values


# <b>Dropping multiple columns where are out of scope for this analysis and/or with 100% null rates:</b>
# 
# REASON, ATLAS BLOCK, TRIP COMMENTS, GROUP IDENTIFIER, IBA CODE, EFFORT AREA HA, BCR CODE, LAST EDITED DATE, PROJECT_CODE,
# HAS MEDIA, TAXONOMIC ORDER, USFWS CODE are out of scope for the project. BREEDING CATEGORY is redundent with BREEDING CODE, COUNTRY CODE is redundent with COUNTRY, STATE CODE is redundent with STATE , COUNTY CODE is redundent with COUNTY, PROTOCOL CODE is redundent with PROTOCOL. SUBSPECIES SCIENTIFIC NAME and SUBSPECIES COMMON NAME are both 100% null.

# In[7]:


raw_data_subset = raw_data.drop(['REASON', 'ATLAS BLOCK', 'TRIP COMMENTS', 'GROUP IDENTIFIER',
                                'IBA CODE', 'SUBSPECIES SCIENTIFIC NAME', 'SUBSPECIES COMMON NAME',
                                'EFFORT AREA HA', 'BCR CODE', 'LAST EDITED DATE', 'HAS MEDIA',
                                'TAXONOMIC ORDER', 'BREEDING CATEGORY', 'SCIENTIFIC NAME', 'COUNTRY CODE', 'Column1',
                                'STATE CODE', 'COUNTY CODE', 'PROTOCOL CODE', 'USFWS CODE', 'PROJECT CODE'], 1)

# replacing whitespace in column headers
raw_data_subset.columns = raw_data_subset.columns.str.replace(' ', '_')
raw_data_subset.head()


# <b>Data type correction: OBSERVATION_DATE</b>

# In[8]:


raw_data_subset['OBSERVATION_DATE'] =  pd.to_datetime(raw_data_subset['OBSERVATION_DATE'],
                                                      format='%m/%d/%Y')
raw_data_subset['DAY'] = raw_data_subset['OBSERVATION_DATE'].dt.day
raw_data_subset['MONTH'] = raw_data_subset.OBSERVATION_DATE.dt.month
raw_data_subset['YEAR'] = raw_data_subset.OBSERVATION_DATE.dt.year


# <b>Addressing messy and missing data in the AGE/SEX column:</b>
# 
# While >97% of rows are null, the data could be valuable for future analysis and deleting rows with missing data would result in removal of otherwise valid data. AGE/SEX data will be cleaned and saved into a separate .csv file to function as a reference table for future analysis where needed. 
# 
# Upon checking unique values in the non-null data, the data appears non-uniform and unstructured. There are multiple birds' age/sex listed in the same entry in many cases. The entries follow the general structure: {sex}, {age group} {count}; {sex}, {age group} {count}; etc... This data will be extracted and each individual age/sex entry will be pivoted into a new table with global_unique_identifier as the index. 

# In[9]:


raw_data_subset['AGE/SEX'].value_counts()


# In[10]:


# splitting the data into a new reference df containing all metadata for age and sex for a given unique identifier, if not null
sex_age_subset = raw_data_subset[['GLOBAL_UNIQUE_IDENTIFIER', 'AGE/SEX']].dropna().reset_index(drop=True)

# need to split the individual values of age/sex into separate rows in a new df on the ; delimiter
sex_age_subset = sex_age_subset.drop('AGE/SEX', axis=1).join(sex_age_subset
                                            ['AGE/SEX'].str.split(';', expand=True).stack().reset_index
                                            (level=1, drop=True).rename('AGE/SEX'))

# extracting the COUNT values into a new column
sex_age_subset['COUNT'] = sex_age_subset['AGE/SEX'].str.extract(pat = '([0-9]+)')
# extracting SEX and AGE into separate columns
sex_age_subset[['SEX','AGE']] = sex_age_subset['AGE/SEX'].str.split(", ",expand=True,)
sex_age_subset["AGE"] = sex_age_subset["AGE"].str.replace("(\([0-9]+\))", "")
# and dropping the original column
sex_age_subset = sex_age_subset.drop(['AGE/SEX'], axis=1)
# saving to a csv to be part of the final dataset for analysis
sex_age_subset.to_csv('Output_Files/age_sex_data_table.csv', index=False)

sex_age_subset.head()


# <b>CATEGORY column has only 1 unique value and provides no useful information.</b>

# In[11]:


print(raw_data_subset['CATEGORY'].value_counts())
raw_data_subset = raw_data_subset.drop(['CATEGORY'], axis=1)
raw_data_subset.head()


# <b>Enriching BREEDING_CODE data:</b>
# 
# This is another case of high null counts, over 96%. However, again, this data is valuable and in-scope of this project. While no reliable inferences could be made about breeding status, the eBird data standard states that a code of (blank) is to be interpretted as not breeding. A lookup table of breeding codes was downloaded from eBird and is used below to enrich the dataset by replacing code values with actual behavioral descriptions.
# 

# In[12]:


unique_breeding_code = pd.unique(raw_data_subset['BREEDING_CODE']) 
breeding_code_lookup = pd.read_csv("Input_Files/breeding_codes.csv")
breeding_code_lookup.head()


# In[13]:


breeding_lookup_dict = pd.Series(breeding_code_lookup.Behavior.values,index=breeding_code_lookup.Code).to_dict()
for key, value in breeding_code_lookup.items():
    key = str(key)

# enrich the working dataframe 
raw_data_subset['BREEDING_CODE'] = raw_data_subset['BREEDING_CODE'].fillna('blank')
raw_data_subset['BREEDING_BEHAVIOR'] = raw_data_subset['BREEDING_CODE'].map(breeding_lookup_dict)  
raw_data_subset = raw_data_subset.drop(['BREEDING_CODE'], axis=1)
raw_data_subset['BREEDING_BEHAVIOR'].value_counts()


# <b>Redundent columns identified: BEHAVIOR_CODE and BREEDING_CODE</b>

# In[14]:


# behavior code is redundent with breeding code -- this column can be dropped
unique_behavior_code = pd.unique(raw_data_subset['BEHAVIOR_CODE'])
raw_data_subset = raw_data_subset.drop(['BEHAVIOR_CODE'], axis=1)
print(unique_behavior_code)


# <b>Filtering out out-of-scope rows:</b>
# 
# Data from the banding dataset in this project is limited to United States. For consistency, filtering for COUNTRY = United States.

# In[15]:


#raw_data_subset =  raw_data_subset[raw_data_subset['COUNTRY'] == 'United States']
#raw_data_subset.head()


# <b>Checking for contiminated data: COMMON_NAME</b>
# 
# Data should only include one species: Piping Plover. Checking for lack of homogeneity/contaminated data.

# In[16]:


# checking to ensure homogeneity in COMMON_NAME field. should only be one version of Piping Plover
unique_common_name = pd.unique(raw_data_subset['COMMON_NAME'])
print(unique_common_name)


# <b>Addressing missing data in the OBSERVATION_COUNT field:</b>
# 
# There are ~11,000 non-numeric entries. Upon inspection, the non-numeric values are all 'X'. eBirders use 'X' to denote an observation for which an exact count was not obtained. Various interpolation methods will be attempted, and one will be selected based on performance. Note that normally outlier analysis would be performed prior to performing interpolation, however it will be bypassed here due to the fact that 1. this data has been vetted by subject matter experts and any unvetted data is omitted from the dataset per eBird and 2. large counts occur during migration events and removal would impact that accuracy.
# 
# The Pandas built-in interpolation methods Linear and Pad will be used, as well as manual interpolation.

# In[17]:


non_numeric_obs_count_rows = raw_data_subset[pd.to_numeric(raw_data_subset['OBSERVATION_COUNT'], errors='coerce').isnull()]
unique_non_numeric_obs_count = pd.unique(non_numeric_obs_count_rows['OBSERVATION_COUNT'])
print(unique_non_numeric_obs_count)
print(len(non_numeric_obs_count_rows))


# In[18]:


# to begin interpolation, converting all 'X' to NaN
raw_data_subset['OBSERVATION_COUNT'] = raw_data_subset['OBSERVATION_COUNT'].replace('X',np.NaN)
# convert column to numeric
raw_data_subset['OBSERVATION_COUNT'] = pd.to_numeric(raw_data_subset['OBSERVATION_COUNT'])
# attempt linear and pad interpolations and compare
raw_data_subset['linear_interp_observation_count'] = raw_data_subset['OBSERVATION_COUNT'].interpolate()
raw_data_subset['pad_interp_observation_count'] = raw_data_subset['OBSERVATION_COUNT'].interpolate(method='pad')


# In[19]:


# also attempting a manual interpolation by averaging OBSERVATION_COUNT over year, month, and state
manual_interpolation_df = raw_data_subset[['YEAR', 'MONTH', 'STATE', 'OBSERVATION_COUNT']]
manual_interpolation_df = manual_interpolation_df.groupby(['YEAR', 'MONTH', 'STATE'])[
    'OBSERVATION_COUNT'].mean().reset_index()
manual_interpolation_df['concatenated_index'] = manual_interpolation_df['YEAR'].astype(str) + manual_interpolation_df[
    'MONTH'].astype(str) + manual_interpolation_df['STATE'].astype(str)
manual_interpolation_df = manual_interpolation_df.set_index('concatenated_index')

# need to temporarily convert the index of the raw_data_subset df to concatenated
raw_data_subset['concatenated_index'] = raw_data_subset['YEAR'].astype(str) + raw_data_subset[
    'MONTH'].astype(str) + raw_data_subset['STATE'].astype(str)
raw_data_subset = raw_data_subset.set_index('concatenated_index')
raw_data_subset['manual_interpolation_observation_count'] = raw_data_subset['OBSERVATION_COUNT'].fillna(manual_interpolation_df['OBSERVATION_COUNT'])

# there are still NaN values after filling since some state/year/month combinations were missing data
# adding a placeholder value of 1
raw_data_subset['manual_interpolation_observation_count'] = raw_data_subset['manual_interpolation_observation_count'].fillna(1)
raw_data_subset = raw_data_subset.reset_index(drop=True)


# In[20]:


# linear and pad interpolations yield similar results.
print('linear interpolation:\n', raw_data_subset['linear_interp_observation_count'].describe(), '\n')
print('pad interpolation:\n', raw_data_subset['pad_interp_observation_count'].describe(), '\n')
print('manual interpolation:\n', raw_data_subset['manual_interpolation_observation_count'].describe(), '\n')
print('original data:\n', raw_data_subset['OBSERVATION_COUNT'].describe(), '\n')


# In[21]:


# scatter plots demonstrating the difference of the interpolated values from the average values by date
counts_comparison_by_date = raw_data_subset[['OBSERVATION_DATE', 'OBSERVATION_COUNT', 'linear_interp_observation_count',
                                             'manual_interpolation_observation_count', 'pad_interp_observation_count']]
obs_cnt_by_date = counts_comparison_by_date.groupby(['OBSERVATION_DATE']).mean()
obs_cnt_by_date['diff_linear'] = obs_cnt_by_date['OBSERVATION_COUNT'] - obs_cnt_by_date['linear_interp_observation_count']
obs_cnt_by_date['diff_pad'] = obs_cnt_by_date['OBSERVATION_COUNT'] - obs_cnt_by_date['pad_interp_observation_count']
obs_cnt_by_date['diff_manual'] = obs_cnt_by_date['OBSERVATION_COUNT'] - obs_cnt_by_date['manual_interpolation_observation_count']

fig1 = px.scatter(obs_cnt_by_date, y = ['diff_linear'], height=300)
fig2 = px.scatter(obs_cnt_by_date, y = ['diff_pad'], height=300)
fig3 = px.scatter(obs_cnt_by_date, y = ['diff_manual'], height=300)
fig1.update_traces(marker=dict(size=4, color='red'))
fig2.update_traces(marker=dict(size=4, color='blue'))
fig3.update_traces(marker=dict(size=4, color='green'))
fig1.show()
fig2.show()
fig3.show()


# <b>Results discussion of OBSERVATION_COUNT analysis:</b>
# 
# The built-in Pandas linear interpolation and the manual interpolation based off State, Year, and Month generated similar results with respect to distance from the calculate averages. The built-in Pandas pad method performed similar but generated additional outliers. Outliers will not be dleeted as observations large migrating flocks were factored into calculations and cannot be ruled out. Moving forward with the manual interpolation.

# In[22]:


from scipy import stats

z_lin = np.abs(stats.zscore(raw_data_subset['linear_interp_observation_count']))
z_pad = np.abs(stats.zscore(raw_data_subset['pad_interp_observation_count']))
z_man = np.abs(stats.zscore(raw_data_subset['manual_interpolation_observation_count']))
threshold = 3
# Position of the outlier
outliers_linear = np.where(z_lin > threshold)
outliers_pad = np.where(z_pad > threshold)
outliers_manual = np.where(z_man > threshold)

linear_outliers_df = raw_data_subset.iloc[outliers_linear]
pad_outliers_df = raw_data_subset.iloc[outliers_pad]
manual_outliers_df = raw_data_subset.iloc[outliers_manual]

print(linear_outliers_df['OBSERVATION_COUNT'].describe())
print(pad_outliers_df['OBSERVATION_COUNT'].describe())
print(manual_outliers_df['OBSERVATION_COUNT'].describe())


# In[23]:


raw_data_subset.drop(['OBSERVATION_COUNT', 'linear_interp_observation_count', 'pad_interp_observation_count'],
                     axis=1, inplace=True)
raw_data_subset.rename(columns={'manual_interpolation_observation_count':'OBSERVATION_COUNT'}, inplace=True)
raw_data_subset.head()


# In[24]:


# checking to ensure homogeneity in LOCALITY_TYPE field. This is a hard-coded part of the submission process.
raw_data_subset['LOCALITY_TYPE'].value_counts()


# In[25]:


# Checking consistency in PROTOCOL_TYPE
raw_data_subset['PROTOCOL_TYPE'].value_counts()


# In[26]:


# Checking consistency in STATE/SUBDIVISION
raw_data_subset['STATE'].value_counts()


# In[27]:


# searching for 'banding' information in SPECIES_COMMENTS
banding_comments = raw_data_subset[raw_data_subset['SPECIES_COMMENTS'].str.contains(" band ", na=False)]
banding_comments.shape[0]


# In[28]:


print(numeric_data)


# <b>Outlier analysis of DURATION_MINUTES column</b>:

# In[29]:


raw_data_subset['DURATION_MINUTES'].describe()


# In[30]:


fig = px.scatter(raw_data_subset,  y = ['DURATION_MINUTES'], height=300)
fig.show()


# In[31]:


# filling missing values with numpy means based on PROTOCOL_TYPE
raw_data_subset['DURATION_MINUTES'] = raw_data_subset.groupby('PROTOCOL_TYPE')['DURATION_MINUTES'].transform(
    lambda grp: grp.fillna(np.mean(grp))
)

raw_data_subset['DURATION_MINUTES'] = raw_data_subset['DURATION_MINUTES'].astype(int)
print(raw_data_subset['DURATION_MINUTES'].isna().sum())


# In[32]:


# 1740 outliers - min of 578 and max of 40011 minutes. Mean of 765 minutes.
z_dur = np.abs(stats.zscore(raw_data_subset['DURATION_MINUTES']))
threshold = 3
# Position of the outlier
outliers_dur = np.where(z_dur > threshold)
dur_outliers_df = raw_data_subset.iloc[outliers_dur]

print(dur_outliers_df['DURATION_MINUTES'].describe())


# In[33]:


# based on basic subject matter expertise - PROTOCOL_TYPE could impact this
# assumption that large parties/special protocols (shorebird surveys, etc.) could have extended observation windows
# let's examine population percentages and max duration of the whole dataset for each protocol type
protocol_type_perc_outliers = dur_outliers_df['PROTOCOL_TYPE'].value_counts(normalize=True) * 100
protocol_type_perc_normal = raw_data_subset['PROTOCOL_TYPE'].value_counts(normalize=True) * 100
protocol_type_mean_normal = raw_data_subset.groupby(['PROTOCOL_TYPE'])['DURATION_MINUTES'].max().sort_values(ascending=False)
print('Population percentages by Protocol Type: outliers only\n', protocol_type_perc_outliers, '\n')
print('Population percentages by Protocol Type: whole dataset\n', protocol_type_perc_normal, '\n')


# Outlier events are limited to Traveling, Stationary, Historical, Area, International Shorebird Survey, and Banding events, with Traveling making the the largest percentage followed at a distance by Stationary, Historical, and Area. The whole dataset has a wider spread of Protocol Types. It may also be useful to get counts of the number of occurrences of '1440' in each of the categories, since 1440 minutes = 24 hours to check if it is prevelant in a given category. 

# In[34]:


protocol_type_subset = raw_data_subset[['PROTOCOL_TYPE', 'DURATION_MINUTES']]
protocol_type_subset_8_hr = protocol_type_subset[protocol_type_subset['DURATION_MINUTES'] >= 480]
protocol_type_subset_8_hr = protocol_type_subset_8_hr.groupby(['PROTOCOL_TYPE']).count().reset_index().rename(columns={'DURATION_MINUTES': 'COUNT'})
protocol_type_subset_12_hr = protocol_type_subset[protocol_type_subset['DURATION_MINUTES'] >= 720]
protocol_type_subset_12_hr = protocol_type_subset_12_hr.groupby(['PROTOCOL_TYPE']).count().reset_index().rename(columns={'DURATION_MINUTES': 'COUNT'})
protocol_type_subset_16_hr = protocol_type_subset[protocol_type_subset['DURATION_MINUTES'] >= 960]
protocol_type_subset_16_hr = protocol_type_subset_16_hr.groupby(['PROTOCOL_TYPE']).count().reset_index().rename(columns={'DURATION_MINUTES': 'COUNT'})
protocol_type_subset_20_hr = protocol_type_subset[protocol_type_subset['DURATION_MINUTES'] >= 1200]
protocol_type_subset_20_hr = protocol_type_subset_20_hr.groupby(['PROTOCOL_TYPE']).count().reset_index().rename(columns={'DURATION_MINUTES': 'COUNT'})
protocol_type_subset_24_hr = protocol_type_subset[protocol_type_subset['DURATION_MINUTES'] >= 1440]
protocol_type_subset_24_hr = protocol_type_subset_24_hr.groupby(['PROTOCOL_TYPE']).count().reset_index().rename(columns={'DURATION_MINUTES': 'COUNT'})


# In[35]:


fig = px.pie(protocol_type_subset_8_hr, values='COUNT', names='PROTOCOL_TYPE', title='Protocol Type for Events >/= 8 Hours')
fig_2 = px.pie(protocol_type_subset_12_hr, values='COUNT', names='PROTOCOL_TYPE', title='Protocol Type for Events >/= 12 Hours')
fig_3 = px.pie(protocol_type_subset_16_hr, values='COUNT', names='PROTOCOL_TYPE', title='Protocol Type for Events >/= 16 Hours')
fig_4 = px.pie(protocol_type_subset_20_hr, values='COUNT', names='PROTOCOL_TYPE', title='Protocol Type for Events >/= 20 Hours')
fig_5 = px.pie(protocol_type_subset_24_hr, values='COUNT', names='PROTOCOL_TYPE', title='Protocol Type for Events >/= 24 Hours')
fig.show()
print('total:', str(sum(protocol_type_subset_8_hr['COUNT'])))
fig_2.show()
print('total:', str(sum(protocol_type_subset_12_hr['COUNT'])))
fig_3.show()
print('total:', str(sum(protocol_type_subset_16_hr['COUNT'])))
fig_4.show()
print('total:', str(sum(protocol_type_subset_20_hr['COUNT'])))
fig_5.show()
print('total:', str(sum(protocol_type_subset_24_hr['COUNT'])))


# As shown above in the pie charts, of the 72 rows with DURATION_MINUTES values greater than or equal to 1440 min (24 hours), 44.4% were Stationary, 50% were Traveling, and 2.78% were Area and Historical, respectively. There is an increase in Stationary events and a reciprocol decrease in Traveling events above the 24 hour mark. Events greater than or equal to 1440 min will be reassigned to the mean of the field, as 24 hour events are highly unlikely. 

# In[36]:


mean_mins = raw_data_subset["DURATION_MINUTES"].mean()
raw_data_subset.loc[(raw_data_subset.DURATION_MINUTES >= 1440),'DURATION_MINUTES']=mean_mins
raw_data_subset['DURATION_MINUTES'].describe()


# In[37]:


fig = px.scatter(raw_data_subset,  y = ['DURATION_MINUTES'], height=300)
fig.show()


# <b>Outlier analysis of EFFORT_DISTANCE_KM column</b>:

# In[38]:


raw_data_subset['EFFORT_DISTANCE_KM'].describe()


# In[39]:


fig = px.scatter(raw_data_subset,  y = ['EFFORT_DISTANCE_KM'], height=300)
fig.show()


# In[40]:


# first we need to ensure data is logical and follows current standards for protocol type descriptions
# specifically, protocol type of stationary should have an effort distance km value of 0.
# not correcting for this will impact filling of na values as well as calculation of outliers
query = """SELECT * FROM  raw_data_subset WHERE PROTOCOL_TYPE = 'Stationary' AND EFFORT_DISTANCE_KM !=0"""
result = ps.sqldf(query, locals())
print(result['EFFORT_DISTANCE_KM'].describe())
result.head()


# In[41]:


# we should replace these values with 0 -- to do this we'll need to set all to 0 in the queried dataframe,
# then use the 'update' function to modify the original dataframe
result['EFFORT_DISTANCE_KM'] = result['EFFORT_DISTANCE_KM'].values[:] = 0.0
raw_data_subset.loc[raw_data_subset.GLOBAL_UNIQUE_IDENTIFIER
                    .isin(result.GLOBAL_UNIQUE_IDENTIFIER), ['EFFORT_DISTANCE_KM']] = result[['EFFORT_DISTANCE_KM']]
result = ps.sqldf(query, locals())
print(result['EFFORT_DISTANCE_KM'].describe())
# all instances of non-zero EFFORT_DISTANCE_KM have been replaced if PROTOCOL_TYPE was set to STATIONARY


# In[42]:


# filling missing values with numpy means based on PROTOCOL_TYPE
raw_data_subset['EFFORT_DISTANCE_KM'] = raw_data_subset.groupby('PROTOCOL_TYPE')['EFFORT_DISTANCE_KM'].transform(
    lambda grp: grp.fillna(np.mean(grp))
)
raw_data_subset['EFFORT_DISTANCE_KM'] = raw_data_subset['EFFORT_DISTANCE_KM'].fillna(0)
print(raw_data_subset['EFFORT_DISTANCE_KM'].isna().sum())


# In[43]:


# 3933 outliers, mean of 35.7km, min of 23.3km, and max of 80.5km.
z_dis = np.abs(stats.zscore(raw_data_subset['EFFORT_DISTANCE_KM']))
threshold = 3
# Position of the outliers
outliers_dis = np.where(z_dis > threshold)
dis_outliers_df = raw_data_subset.iloc[outliers_dis]

print(dis_outliers_df['EFFORT_DISTANCE_KM'].describe())


# In[44]:


# based on basic subject matter expertise - PROTOCOL_TYPE could impact this value as well
# assumption that large parties/special protocols (shorebird surveys, etc.) could have extended distances
# let's examine population percentages and max duration of the whole dataset for each protocol type
protocol_type_perc_outliers_dis = dis_outliers_df['PROTOCOL_TYPE'].value_counts(normalize=True) * 100
protocol_type_perc_normal_dis = raw_data_subset['PROTOCOL_TYPE'].value_counts(normalize=True) * 100
protocol_type_mean_normal_dis = raw_data_subset.groupby(['PROTOCOL_TYPE'])['EFFORT_DISTANCE_KM'].max().sort_values(ascending=False)
print('Population percentages by Protocol Type: outliers only\n', protocol_type_perc_outliers_dis, '\n')
print('Population percentages by Protocol Type: whole dataset\n', protocol_type_perc_normal_dis, '\n')


# It is not immediately obvious that the outlier EFFORT_DISTANCE_KM values were erronious opposed to legitimate entries, as 80km traveling distances are entirely possible during motor vehicle trips. The outliers will remain in the dataset for this field.

# <b>Outlier analysis of NUMBER_OBSERVERS column:</b>

# In[45]:


fig = px.scatter(raw_data_subset,  y = ['NUMBER_OBSERVERS'], height=300)
fig.show()
print(raw_data_subset['NUMBER_OBSERVERS'].describe())


# In[46]:


# filling missing values with numpy means based on PROTOCOL_TYPE and OBSERVER_ID, else 0
raw_data_subset['NUMBER_OBSERVERS'] = raw_data_subset.groupby(['PROTOCOL_TYPE', 'OBSERVER_ID'])['NUMBER_OBSERVERS'].transform(
    lambda grp: grp.fillna(np.mean(grp))
)
raw_data_subset['NUMBER_OBSERVERS'] = raw_data_subset['NUMBER_OBSERVERS'].fillna(1) # cannot have 0 observers
print(raw_data_subset['NUMBER_OBSERVERS'].isna().sum())


# In[47]:


# 78 outliers, mean of 273.3, min of 75, and max of 9323.
z_dis = np.abs(stats.zscore(raw_data_subset['NUMBER_OBSERVERS']))
threshold = 3
# Position of the outliers
outliers_nobs = np.where(z_dis > threshold)
nobs_outliers_df = raw_data_subset.iloc[outliers_nobs]

print(nobs_outliers_df['NUMBER_OBSERVERS'].describe())


# In[48]:


# based on basic subject matter expertise - again, PROTOCOL_TYPE may have an impact on this value
# assumption that special protocols (surveys, competitions) could have increased participants 
# let's examine population percentages and max duration of the whole dataset for each protocol type
protocol_type_perc_outliers_nobs = nobs_outliers_df['PROTOCOL_TYPE'].value_counts(normalize=True) * 100
protocol_type_perc_normal_nobs = raw_data_subset['PROTOCOL_TYPE'].value_counts(normalize=True) * 100
protocol_type_mean_normal_nobs = raw_data_subset.groupby(['PROTOCOL_TYPE'])['NUMBER_OBSERVERS'].max().sort_values(ascending=False)
print('Population percentages by Protocol Type: outliers only\n', protocol_type_perc_outliers_nobs, '\n')
print('Population percentages by Protocol Type: whole dataset\n', protocol_type_perc_normal_nobs, '\n')


# In[49]:


# there are no special protocols in the outlier counts - only Traveling, Stationary, Incidental, and Area
# the outlier population is small and significantly different from the rest of the dataset
# for simplicity the outliers will be transformed to the mean of the values
raw_data_subset.loc[raw_data_subset.GLOBAL_UNIQUE_IDENTIFIER
                    .isin(nobs_outliers_df.GLOBAL_UNIQUE_IDENTIFIER), ['NUMBER_OBSERVERS']] = np.nan
mean_obs = raw_data_subset['NUMBER_OBSERVERS'].mean()
raw_data_subset['NUMBER_OBSERVERS'].fillna(mean_obs, inplace=True)
raw_data_subset['NUMBER_OBSERVERS'] = raw_data_subset['NUMBER_OBSERVERS'].astype(int)


# In[50]:


fig = px.scatter(raw_data_subset,  y = ['NUMBER_OBSERVERS'], height=300)
fig.show()
print(raw_data_subset['NUMBER_OBSERVERS'].describe())


# In[51]:


# all outliers and nan values have been encounted for in numeric fields
print(raw_data_subset.describe())


# In[52]:


raw_data_subset.head()


# In[53]:


raw_data_subset.to_csv('Output_Files/cleaned_ebird_data_output.csv', index=False)


# In[ ]:




