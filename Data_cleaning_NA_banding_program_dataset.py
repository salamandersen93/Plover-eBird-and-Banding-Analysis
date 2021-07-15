#!/usr/bin/env python
# coding: utf-8

# Construction of curated bird banding dataset from https://www.sciencebase.gov/catalog/item/60914db3d34e791692e13a22
# 
# Species being targeted in this dataset construction are selected species of plover: Piping Plover, Black-bellied Plover, Wilson's Plover, Semipalmated Plover, and Snowy Plover. The datasets available for download are split into 10 groups based on taxonomic order. All plovers being targeted in this dataset construction were contained within dataset 4. Data was filtered using powerquery in Excel to generate a subset of the data only containing desired species. 
# 
# Sciencebase provides a series of reference datasets/lookup tables to accompany the datasets, which contain valuable metadata associated with codes used in the primary dataset. These lookup tables will be leveraged to enrich the extracted subset of data and generate a more complete table, which will include well-labeled metadata.
# 
# Cleaning procedures will be performed on the dataset including: standardizing casing across strings + removing whitespace and newlines in select columns, correcting typos, standardizing dates, removing invalid data, and addressing missing data. 

# In[1]:


import pandas as pd
import glob


# In[2]:


# get raw data -- note common forms of N/A are being addressed while reading into the dataframe
missing_values = ["n/a", "na", "N/A", "NA" ,"--", "-"]
raw_data = pd.read_csv("Input_Files/raw_data_plover_banding.csv", na_values = missing_values)
pd.set_option('display.max_columns', None) # lots of columns, need to set to display max
raw_data.head()


# In[3]:


# several columns use codes in reference to values in lookup tables
# some interpreted as floats, so changing all to strings and removing tailing decimals
convert_age_dict = {'BIRD_STATUS': str,
                    'EXTRA_INFO_CODE': str,
                    'AGE_CODE': str,
                    'SEX_CODE': str,
                    'BAND_STATUS_CODE': str,
                    'COORDINATES_PRECISION_CODE': str,
                    'SPECIES_ID': str,
                    'HOW_OBTAINED_CODE': str,
                    'WHO_OBTAINED_CODE': str,
                    'REPORTING_METHOD_CODE': str,
                    'PRESENT_CONDITION_CODE': str,
                    'MIN_AGE_AT_ENC': str       
               }
working_df = raw_data.astype(convert_age_dict, errors = 'ignore').replace('\.0', '', regex=True)


# In[4]:


# trimming whitespace from beginning and end of each string column
# note that any case standardization will be performed AFTER enrichment of the primary dataset using the reference tables
working_df = working_df.applymap(lambda x: x.strip() if isinstance(x, str) else x)


# In[5]:


# get reference tables to be used to enrich primary dataset - will iterate through each table to understand context and structure
# as well as which fields can be added to the primary dataset 
filenames = glob.glob("Input_Files/lookup_tables/*.csv")
d = {}
for x in filenames:
    print(x)
    adj_key = x.replace("Input_Files/lookup_tables\\", '')
    adj_key = adj_key.replace('.csv', '')
    d[adj_key] = pd.read_csv(x)

# print list of reference tables -- determine which will be used to translate codes and enrich the primary dataset
for key, value in d.items():
    print(key)


# In[6]:


#'country_state' not needed as it is already captured in ISO format in the primary dataset
# inexact date also not needed as event_date is provided along with day, month, year -- more on date standardization later

# note that reference datasets are not uniform in ontology/header nomenclature, so a function-based approach may produce errors
# a similar approach will be used for each reference table, with minor variations:
# 1. dataframe will be created from the reference table and visualized to understand the context of the data
# 2. for each each field deemed valuable to be added as metadata to the primary table, a dictionary will be created
#    using the CODE as the key and the target field to be added as the value
# 3. the map functionality in pandas will be used to create a new column in the primary dataset containing the new data.
#    also note some data type transformations/whitespace  are necessary

# now iterating through each of the reference table, starting with age
# note a function-based approach would be erronious as ontology/column header nomenclature in reference tables is not uniform 
age_df = d['age']
age_df['AGE_CODE'] = age_df['AGE_CODE'].astype(str)
age_df.head()


# In[7]:


# can use 'age_description' column to enrich the primary dataset
age_lookup = pd.Series(age_df.AGE_DESCRIPTION.values,index=age_df.AGE_CODE).to_dict()
for key, value in age_lookup.items():
    key = str(key)

# enrich the working dataframe 
working_df['AGE'] = working_df['AGE_CODE'].map(age_lookup)  
working_df.head()


# In[8]:


# now band_status
band_status_df = d['band_status']
band_status_df.head()


# In[9]:


band_status_lookup = pd.Series(band_status_df.BAND_STATUS_DESCRIPTION.values,index=band_status_df.BAND_STATUS_CODE).to_dict()
for key, value in band_status_lookup.items():
    key = str(key)

# enrich the working dataframe 
working_df['BAND_STATUS'] = working_df['BAND_STATUS_CODE'].map(band_status_lookup)  
working_df.head()


# In[10]:


# now band type
band_type_df = d['band_type']
band_type_df.head()


# In[11]:


# for band type there are two fields that should be extracted and captured in the primary dataset, description and closure type
band_type_desc_lookup = pd.Series(band_type_df.BAND_TYPE_DESCRIPTION.values,index=band_type_df.BAND_TYPE_CODE).to_dict()
for key, value in band_status_lookup.items():
    key = str(key)
    
band_type_clos_lookup = pd.Series(band_type_df.BAND_CLOSURE.values,index=band_type_df.BAND_TYPE_CODE).to_dict()
for key, value in band_status_lookup.items():
    key = str(key)

# enrich the working dataframe 
working_df['BAND_TYPE'] = working_df['BAND_TYPE_CODE'].map(band_type_desc_lookup)
working_df['BAND_CLOSURE_TYPE'] = working_df['BAND_TYPE_CODE'].map(band_type_clos_lookup)  
working_df.head()


# In[12]:


#now bird status
bird_status_df = d['bird_status']
bird_status_df.head()


# In[13]:


bird_status_lookup = pd.Series(bird_status_df.BIRD_STATUS_DESCRIPTION.values,index=bird_status_df.BIRD_STATUS).to_dict()
for key, value in bird_status_lookup.items():
    key = str(key)

# enrich the working dataframe - note lack of word 'code' in working dataset column name "bird_status" rather than "bird_status_code"
working_df['BIRD_STATUS_DESCRIPTION'] = working_df['BIRD_STATUS'].map(bird_status_lookup)  
working_df.head()


# In[14]:


# now coordinates_precision
coordinates_precision_df = d['coordinates_precision']
coordinates_precision_df.head()


# In[15]:


coordinates_precision_lookup = pd.Series(coordinates_precision_df.COORDINATES_PRECISION_DESC.values,
                                         index=coordinates_precision_df.COORDINATES_PRECISION_CODE).to_dict()

for key, value in coordinates_precision_lookup.items():
    key = int(key)

# enrich the working dataframe
working_df['COORDINATES_PRECISION_CODE'] = working_df['COORDINATES_PRECISION_CODE'].astype(int)
working_df['COORDINATES_PRECISION'] = working_df['COORDINATES_PRECISION_CODE'].map(coordinates_precision_lookup)  
working_df.head()


# In[16]:


# now event type
event_type_df = d['event_type']
event_type_df.head()


# In[17]:


event_type_lookup = pd.Series(event_type_df.EVENT_TYPE_DESC.values,index=event_type_df.EVENT_TYPE).to_dict()

for key, value in event_type_lookup.items():
    key = str(key)

# enrich the working dataframe 
working_df['EVENT_TYPE_DESCRIPTION'] = working_df['EVENT_TYPE'].map(event_type_lookup)
working_df.head()


# In[18]:


# now extra info
extra_info_df = d['extra_info']
extra_info_df.head()


# In[19]:


extra_info_lookup = pd.Series(extra_info_df.EXTRA_INFO_CODE_DESCRIPTION.values,index=extra_info_df.EXTRA_INFO_CODE).to_dict()

for key, value in extra_info_lookup.items():
    key = str(key)

# enrich the working dataframe 
working_df['EXTRA_INFO'] = working_df['EXTRA_INFO_CODE'].map(extra_info_lookup)
working_df.head()


# In[20]:


# now how_obtained
how_obtained_df = d['how_obtained']
how_obtained_df['HOW_OBTAINED_CODE'] = how_obtained_df['HOW_OBTAINED_CODE'].astype(str)
how_obtained_df.head()


# In[21]:


how_obtained_lookup = pd.Series(how_obtained_df.HOW_OBTAINED_DESCRIPTION.values,
                                index=how_obtained_df.HOW_OBTAINED_CODE).to_dict()

for key, value in how_obtained_lookup.items():
    key = str(key)
    
# enrich the working dataframe
working_df['HOW_OBTAINED_CODE'] = working_df['HOW_OBTAINED_CODE'].astype(int, errors='ignore')
working_df['HOW_OBTAINED'] = working_df['HOW_OBTAINED_CODE'].map(how_obtained_lookup)  
working_df.head()


# In[22]:


# QA check to ensure all codes were interpretted correctly and no values were missed
n = len(pd.unique(working_df['HOW_OBTAINED']))
n1 = len(pd.unique(working_df['HOW_OBTAINED_CODE']))
list_unique_all = pd.unique(working_df['HOW_OBTAINED']) 
list_unique_all_codes = pd.unique(working_df['HOW_OBTAINED_CODE']) 
print(n)
print(list_unique_all)
print(n1)
print(list_unique_all_codes)


# In[23]:


# now present condition
present_condition_df = d['present_condition']
present_condition_df['PRESENT_CONDITION_CODE'] = present_condition_df['PRESENT_CONDITION_CODE'].astype(str)
present_condition_df.head()


# In[24]:


# extracting two fields - 'PRESENT_CONDITION_BIRD' and 'PRESENT_CONDITION_BAND'
present_condition_bird_lookup = pd.Series(present_condition_df.PRESENT_CONDITION_BIRD.values,
                                          index=present_condition_df.PRESENT_CONDITION_CODE).to_dict()

for key, value in present_condition_bird_lookup.items():
    key = str(key)
    
present_condition_band_lookup = pd.Series(present_condition_df.PRESENT_CONDITION_BAND.values,
                                          index=present_condition_df.PRESENT_CONDITION_CODE).to_dict()

for key, value in present_condition_band_lookup.items():
    key = str(key)

# enrich the working dataframe 
working_df['PRESENT_CONDITION_BIRD'] = working_df['PRESENT_CONDITION_CODE'].map(present_condition_bird_lookup)
working_df['PRESENT_CONDITION_BAND'] = working_df['PRESENT_CONDITION_CODE'].map(present_condition_band_lookup)
working_df.head()


# In[25]:


# quick QA check to make sure the last two integrations worked since all head values are NaN
n = len(pd.unique(working_df['PRESENT_CONDITION_BIRD']))
n2 = len(pd.unique(working_df['PRESENT_CONDITION_BAND']))
n3 = len(pd.unique(working_df['PRESENT_CONDITION_CODE']))
list_unique_all_bird = pd.unique(working_df['PRESENT_CONDITION_BIRD']) 
list_unique_all_band = pd.unique(working_df['PRESENT_CONDITION_BAND'])
print(n)
print(n2)
print(n3)
print(list_unique_all_bird)
print(list_unique_all_band)


# In[26]:


# now record_source
record_source_df = d['record_source']
record_source_df.head()


# In[27]:


record_source_lookup = pd.Series(record_source_df.EVENT_TYPE_DESC.values,index=record_source_df.EVENT_TYPE).to_dict()

for key, value in record_source_lookup.items():
    key = str(key)

# enrich the working dataframe 
working_df['RECORD_SOURCE_DESCRIPTION'] = working_df['RECORD_SOURCE'].map(record_source_lookup)
working_df.head()


# In[28]:


# now reporting_method
reporting_method_df = d['reporting_method']
reporting_method_df.head()


# In[29]:


reporting_method_lookup = pd.Series(reporting_method_df.REPORTING_METHOD_DESC.values,
                                    index=reporting_method_df.REPORTING_METHOD_CODE.astype(str)).to_dict()

# enrich the working dataframe 
working_df['REPORTING_METHOD'] = working_df['REPORTING_METHOD_CODE'].map(reporting_method_lookup)
working_df.head()


# In[30]:


# another QA check since all in head are NaN
n = len(pd.unique(working_df['REPORTING_METHOD']))
n1 = len(pd.unique(working_df['REPORTING_METHOD_CODE']))
list_unique_all_code = pd.unique(working_df['REPORTING_METHOD_CODE']) 
list_unique_all_values = pd.unique(working_df['REPORTING_METHOD'])
print(n)
print(n1)
print(list_unique_all_code)
print(list_unique_all_values)


# In[31]:


# now sex
sex_df = d['sex']
sex_df.head()


# In[32]:


sex_lookup = pd.Series(sex_df.SEX_DESCRIPTION.values,index=sex_df.SEX_CODE.astype(str)).to_dict()

# sex_code 6 and 7 are not necessary - should be limited to male, female, and unknown
working_df['SEX_CODE'] = working_df['SEX_CODE'].replace('6', '4')
working_df['SEX_CODE'] = working_df['SEX_CODE'].replace('7', '5')

# enrich the working dataframe 
working_df['SEX'] = working_df['SEX_CODE'].map(sex_lookup)

print(working_df['SEX'].describe())
working_df.head()


# In[33]:


#species
species_df = d['species']
species_df.head()


# In[34]:


# taking species_name, alpha_code, taxonomic_order, endangered, and allowablesize from the species DF
species_name_lookup = pd.Series(species_df.SPECIES_NAME.values,
                                    index=species_df.SPECIES_ID.astype(str)).to_dict()
species_alpha_lookup = pd.Series(species_df.ALPHA_CODE.values,
                                    index=species_df.SPECIES_ID.astype(str)).to_dict()
species_tax_order_lookup = pd.Series(species_df.TAXONOMIC_ORDER.values,
                                    index=species_df.SPECIES_ID.astype(str)).to_dict()
species_endangered_lookup = pd.Series(species_df.ENDANGERED.values,
                                    index=species_df.SPECIES_ID.astype(str)).to_dict()
species_allowable_size_lookup = pd.Series(species_df.ALLOWABLESIZE.values,
                                    index=species_df.SPECIES_ID.astype(str)).to_dict()

# enrich the working dataframe 
working_df['SPECIES_NAME'] = working_df['SPECIES_ID'].map(species_name_lookup)
working_df['SPECIES_ALPHA_CODE'] = working_df['SPECIES_ID'].map(species_alpha_lookup)
working_df['TAXONOMIC_ORDER'] = working_df['SPECIES_ID'].map(species_tax_order_lookup)
working_df['ENDANGERED'] = working_df['SPECIES_ID'].map(species_endangered_lookup)
working_df['ALLOWABLE_BAND_SIZE'] = working_df['SPECIES_ID'].map(species_allowable_size_lookup)
working_df.head()


# In[35]:


#who_obtained
who_obtained_df = d['who_obtained']
who_obtained_df.head()


# In[36]:


who_obtained_lookup = pd.Series(who_obtained_df.WHO_OBTAINED_DESCRIPTION.values,
                                index=who_obtained_df.WHO_OBTAINED_CODE.astype(str)).to_dict()

# enrich the working dataframe 
working_df['WHO_OBTAINED'] = working_df['WHO_OBTAINED_CODE'].map(who_obtained_lookup)
working_df.head()


# In[37]:


# another QA check since all in head are NaN
n = len(pd.unique(working_df['WHO_OBTAINED']))
list_unique_all = pd.unique(working_df['WHO_OBTAINED_CODE']) 
print(n)
print(list_unique_all)


# In[38]:


# selection of columns from the current working dataframe to transfer to a consolidated dataframe
# printing complete list of columns
working_df.columns


# In[39]:


working_df_subset = working_df[['BAND', 'ORIGINAL_BAND', 'OTHER_BANDS', 'BAND_TYPE', 'BAND_STATUS', 'PRESENT_CONDITION_BAND',
                               'BAND_CLOSURE_TYPE', 'EVENT_TYPE_DESCRIPTION', 'EVENT_DATE', 'EVENT_DAY',
                               'EVENT_MONTH', 'EVENT_YEAR', 'ISO_COUNTRY', 'ISO_SUBDIVISION', 'LAT_DD', 'LON_DD',
                               'COORDINATES_PRECISION', 'PERMIT', 'HOW_OBTAINED', 'WHO_OBTAINED','REPORTING_METHOD',
                               'SPECIES_ID', 'SPECIES_NAME', 'BIRD_STATUS_DESCRIPTION',
                               'AGE', 'SEX', 'MIN_AGE_AT_ENC', 'SPECIES_ALPHA_CODE', 'TAXONOMIC_ORDER', 'ENDANGERED',
                               'ALLOWABLE_BAND_SIZE', 'PRESENT_CONDITION_BIRD', 'EXTRA_INFO', 'RECORD_SOURCE_DESCRIPTION']].copy()
working_df_subset.head()


# In[40]:


# next step - harmonization of ontologies within species column
# identify inconsistencies in spelling, spacing, format, duplicate errors, and invalid data


# In[41]:


# start with species_name
counts = working_df_subset.groupby(['SPECIES_NAME', 'SPECIES_ID'])['BAND'].nunique()
print(counts)


# In[42]:


# recall the only species used in this dataset were 'Black-bellied Plover', 'Piping Plover', 
# 'Semipalmated Plover', 'Snowy Plover', and 'Wilson's Plover'
# Common Tern and Long-billed Dowitcher do not belong, and are likely the product of a transcription error in species_ID
# in the raw dataset during data entry... these rows will be dropped from the dataset.

working_df_subset = working_df_subset[working_df_subset.SPECIES_NAME != 'Long-billed Dowitcher']
working_df_subset = working_df_subset[working_df_subset.SPECIES_NAME != 'Common Tern']

counts = working_df_subset.groupby(['SPECIES_NAME', 'SPECIES_ID'])['BAND'].nunique()
print(counts)


# In[43]:


# get unique values of day, month, year fields to ensure no abberant data or invalid data
days_unique = pd.unique(working_df['EVENT_DAY']) 
month_unique = pd.unique(working_df['EVENT_MONTH']) 
years_unique = pd.unique(working_df['EVENT_YEAR']) 
working_df_subset.drop(working_df_subset[working_df_subset['EVENT_MONTH'] > 12].index, inplace = True)
month_unique_fixed = pd.unique(working_df_subset['EVENT_MONTH']) 

print(days_unique)
print(month_unique) #obvious human error in the dataset - month '83' - this needs to be removed. will remove any values > 12.
print(years_unique)
print(month_unique_fixed)


# In[44]:


# next step of process - counting and dealing with missing data
# get full number of rows for context

index = working_df_subset.index
number_of_rows = len(index) 
print('TOTAL ROWS:', number_of_rows)
print(working_df_subset.isna().sum())


# In[45]:


# for some columns like BAND_STATUS, HOW_OBTAINED, WHO_OBTAINED, REPORTING_METHOD, BIRD_STATUS_DESCRIPTION, AGE, 
# SEX, PRESENT_BIRD_CONDITION there are expected values, and the lack of a value indicates that the value is not known
# or was missed during data entry, hence we can replace missing values with "unknown"
working_df_subset[['BAND_STATUS', 'HOW_OBTAINED', 'WHO_OBTAINED', 'REPORTING_METHOD', 'BIRD_STATUS_DESCRIPTION',
                  'AGE', 'SEX', 'PRESENT_CONDITION_BIRD', 'PRESENT_CONDITION_BAND']] = working_df_subset[['BAND_STATUS', 'HOW_OBTAINED', 'WHO_OBTAINED',
                                                                                'REPORTING_METHOD', 'BIRD_STATUS_DESCRIPTION',
                                                                                'AGE', 'SEX', 'PRESENT_CONDITION_BIRD',
                                                                                'PRESENT_CONDITION_BAND']].fillna(value='Unknown')
print(working_df_subset.isna().sum())


# In[46]:


# however for columns like PERMIT, OTHER_BANDS, ENDANGERED, AND ISO_SUBDIVISION, the missing data could also indicate a negative
# or false value (e.g. no permit, no other bands, or no ISO subdivision) -- we can check the unique values to determine if action is needed

other_bands_unique = pd.unique(working_df['OTHER_BANDS']) 
iso_subdivision_unique = pd.unique(working_df['ISO_SUBDIVISION']) 
permit_unique = pd.unique(working_df['PERMIT']) 
endangered_unique = pd.unique(working_df['ENDANGERED']) 

print(other_bands_unique) # Na values should be left as is - values are band identifiers 
print(iso_subdivision_unique) # NA values should be left as is - values are reserved for US states
# or canadian provinces and are not applicable to other countries
print(permit_unique) # NA values should be left as is - values are permit identifiers
print(endangered_unique) # Needs to be analyzed further.


# In[47]:


# As shown above, the ENDANGERED field should be converted to boolean. Unique values are either nan or Y.
# We cannot trust that all nan values are equivalent to False, however. 
# To analyze, let us print all rows where species are listed as endangered. 
# Subject matter expertise would ideally be required to determine whether the endangered status referenced to local status

endangered_species_country_subdiv = working_df_subset.groupby(['ENDANGERED', 'SPECIES_NAME',
                                                              'ISO_COUNTRY', 'ISO_SUBDIVISION']).size().reset_index(name='Freq')
endangered_species_country_subdiv.head(100)

# the dataset shown below gives us all instances of where species are listed as endangered.
# this shows us that piping plover and western snowy plover are two species considered endangered in certain locations
# without subject matter expertise from the dataset owner, best to leave as is


# In[48]:


# also note there are 2 dates missing in the EVENT_DATE field -- we can check to see if EVENT_DAY, EVENT_MONTH, and EVENT_YEAR
# are valid data points so we can reconstruct the EVENT_DATE

na_dates = working_df_subset[working_df_subset['EVENT_DATE'].isna()]
na_dates.head()


# In[49]:


# can fix this using the datetime module with format='%Y%j
na_dates['EVENT_DATE'] = pd.to_datetime(na_dates['EVENT_YEAR'] * 1000 + na_dates['EVENT_DAY'], format='%Y%j')
na_dates['EVENT_DATE'] = pd.to_datetime(na_dates['EVENT_DATE']).dt.strftime('%m/%d/%Y')
na_dates.head()


# In[50]:


# removing impacted rows from primary dataframe then re-adding from the subset copy using BAND as unique ID
# first we need to make sure there are no duplicate BAND values so no additional rows get deleted
working_df_subset.BAND.duplicated().sum 


# In[51]:


bands_to_correct_date = na_dates['BAND'].tolist()
working_df_subset = working_df_subset[~working_df_subset['BAND'].isin(bands_to_correct_date)]
working_df_subset.append(na_dates)


# In[52]:


# convert the entire column to datetime
working_df_subset['EVENT_DATE'] = pd.to_datetime(working_df_subset['EVENT_DATE'], format= '%m/%d/%Y')


# In[53]:


# one more check of datetypes to ensure types are correct
working_df_subset.dtypes


# In[54]:


# note the taxonomic order is a float - should be an integer as it is represented as a whole number
working_df_subset['TAXONOMIC_ORDER'] = working_df_subset['TAXONOMIC_ORDER'].astype(int)
working_df_subset.head()


# In[55]:


# case standardization - lower
working_df_subset['BAND_TYPE'] = working_df_subset['BAND_TYPE'].str.lower()
working_df_subset['BAND_STATUS'] = working_df_subset['BAND_STATUS'].str.lower()
working_df_subset['PRESENT_CONDITION_BAND'] = working_df_subset['PRESENT_CONDITION_BAND'].str.lower()
working_df_subset['EVENT_TYPE_DESCRIPTION'] = working_df_subset['EVENT_TYPE_DESCRIPTION'].str.lower()
working_df_subset['COORDINATES_PRECISION'] = working_df_subset['COORDINATES_PRECISION'].str.lower()
working_df_subset['HOW_OBTAINED'] = working_df_subset['HOW_OBTAINED'].str.lower()
working_df_subset['WHO_OBTAINED'] = working_df_subset['WHO_OBTAINED'].str.lower()
working_df_subset['REPORTING_METHOD'] = working_df_subset['REPORTING_METHOD'].str.lower()
working_df_subset['SPECIES_NAME'] = working_df_subset['SPECIES_NAME'].str.lower()
working_df_subset['BIRD_STATUS_DESCRIPTION'] = working_df_subset['BIRD_STATUS_DESCRIPTION'].str.lower()
working_df_subset['AGE'] = working_df_subset['AGE'].str.lower()
working_df_subset['SEX'] = working_df_subset['SEX'].str.lower()
working_df_subset['ENDANGERED'] = working_df_subset['ENDANGERED'].str.lower()
working_df_subset['PRESENT_CONDITION_BIRD'] = working_df_subset['PRESENT_CONDITION_BIRD'].str.lower()
working_df_subset['EXTRA_INFO'] = working_df_subset['EXTRA_INFO'].str.lower()


# case standardization - upper
working_df_subset['BAND'] = working_df_subset['BAND'].str.upper()
working_df_subset['ORIGINAL_BAND'] = working_df_subset['ORIGINAL_BAND'].str.upper()
working_df_subset['OTHER_BANDS'] = working_df_subset['OTHER_BANDS'].str.upper()
working_df_subset['BAND_CLOSURE_TYPE'] = working_df_subset['BAND_CLOSURE_TYPE'].str.upper()
working_df_subset['ISO_COUNTRY'] = working_df_subset['ISO_COUNTRY'].str.upper()
working_df_subset['ISO_SUBDIVISION'] = working_df_subset['ISO_SUBDIVISION'].str.upper()
working_df_subset['ALLOWABLE_BAND_SIZE'] = working_df_subset['ALLOWABLE_BAND_SIZE'].str.upper()


# In[56]:


working_df_subset.to_csv('Output_Files/cleaned_banding_data_output.csv', index=False)


# In[ ]:




