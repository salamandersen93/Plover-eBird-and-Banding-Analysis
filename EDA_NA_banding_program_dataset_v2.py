#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install pandasql')


# In[2]:


import pandas as pd
import pandasql as ps
import numpy as np
import seaborn as sns
import plotly.express as px


# In[3]:


cleaned_data = pd.read_csv("cleaned_data_output.csv")
pd.set_option('display.max_columns', None) # lots of columns, need to set to display max
cleaned_data.head()


# <b>visualizing the relationship between species banded and year</b>:
# Taking a look at conservation efforts of the various plover species over time. Piping plover has been a species of interest due to threatened habitat and population loss, so it can be expected that banding efforts also have increased over time. 

# In[4]:


def bar_chart_viz_counts (df_in, field_1, field_2, title, show_text, legend):
    
    if show_text == True:
        fig = px.bar(df_in, x=field_1, y ="COUNT", color=field_2, hover_data=['COUNT'], barmode='stack', text='COUNT')
    else:
        fig = px.bar(df_in, x=field_1, y ="COUNT", color=field_2, hover_data=['COUNT'], barmode='stack')
        
    if legend is not False:
        fig.update_layout(legend=legend)
        
    fig.update_layout(title=title,
                 xaxis=dict(title=field_1, tickmode='linear'))
    
    fig.show()


# In[5]:


# visualizing relationship of species banded over time
species_name_counts = cleaned_data.SPECIES_NAME.value_counts()
agg_data_species_year = cleaned_data.groupby(['SPECIES_NAME', 'EVENT_YEAR'], 
                                             as_index=False)['BAND'].count().rename(columns={'BAND' : 'COUNT'})

bar_chart_viz_counts(agg_data_species_year, 'EVENT_YEAR', 'SPECIES_NAME', 'BANDING EVENTS BY YEAR AND SPECIES', False, False)


# <b>Regression analysis of piping plover banding</b>:
# Based on the bar chart above, there does indeed appear to be an increase in banding of piping plovers over time beginning in the 1980's. This aligns with information from the Fish and Wildlife Service in the link below. A regression analysis is executed below to demonstrate the trend across species.
# 
# https://www.fws.gov/midwest/endangered/pipingplover/piplconservation.html

# In[6]:


# frequency of species banding by year
fig = px.scatter(agg_data_species_year, x='EVENT_YEAR', y='COUNT', color='SPECIES_NAME', trendline="ols")
fig.show()

results = px.get_trendline_results(fig)
results.query("SPECIES_NAME == 'piping plover'").px_fit_results.iloc[0].summary()


# <b>Assessing relationship between US State and banding efforts - is there a geographical connection?</b>
# <br>
# Let us specifically focus efforts on piping plover, due to the increase in banding effort over the past 40 years. Has the uptick in banding been connected to specific states within the US? 

# In[7]:


species_date_and_state = cleaned_data.groupby(['SPECIES_NAME', 'EVENT_YEAR', 'ISO_SUBDIVISION'], 
                                             as_index=False)['BAND'].count().rename(columns={'BAND' : 'COUNT'})
piping_plover_US_dates = ps.sqldf("SELECT * FROM species_date_and_state                                   WHERE SPECIES_NAME = 'piping plover' AND                                   ISO_SUBDIVISION LIKE 'US-%'")
piping_plover_US_dates


# <b>Visualizing the top 10 states with the highest Piping Plover banding counts</b>:
# First, the nlargest are selected from the queried dataframe above, then visualized using a line plot. While there is an positive in most of the states, the following states stand out: ND, NE, and MI stand out in terms of AOC. A quick google search confirms the efforts in these states. The breeding range of piping plovers turns out to be in ND and NE, while Mississippi hosts critical habitat for the winter non-breeding population.
# https://gf.nd.gov/wildlife/id/shorebirds/piping-plover
# https://platteriverprogram.org/target-species/piping-plover
# https://www.fws.gov/midwest/endangered/pipingplover/pdf/CCSpiplNoApp2012.pdf

# In[8]:


def line_chart_viz_counts (df_in, field_x, field_y, field_color, title, legend):
    
    fig = px.line(df_in,
                  x=field_x,
                  y=field_y,  
                  color=field_color)
    
    if legend != False:
        fig.update_layout(title=title,
                        showlegend=True,
                        yaxis={"visible":True})
    else:
        fig.update_layout(title=title,
                showlegend=False,
                yaxis={"visible":True})
    
    fig.show()


# In[9]:


state_counts = piping_plover_US_dates.groupby(['ISO_SUBDIVISION'])['COUNT'].sum()
top_10_states = state_counts.nlargest(10)
top_10_states = top_10_states.to_frame().reset_index()
top_10_states

top_10_list = top_10_states['ISO_SUBDIVISION'].tolist()
top_10_states_piping_plover_df = piping_plover_US_dates.query('ISO_SUBDIVISION in @top_10_list')

line_chart_viz_counts (top_10_states_piping_plover_df, "EVENT_YEAR", "COUNT", "ISO_SUBDIVISION",
                       "TOP 10 STATES - PIPING PLOVER EVENTS", True)


# <b>Assessing the relationship between event month, banding frequency, and state</b>:<br>
# The visualization below indicates that most banding occurs in the summer months. This aligns perfectly with the geographical data -- plovers are in their breeding territories during summer. Observation coutns are highest for the summer months, when the birds are in breeding territory.

# In[10]:


agg_data_species_event_month = cleaned_data.groupby(['SPECIES_NAME', 'EVENT_MONTH'], 
                                             as_index=False)['BAND'].count().rename(columns={'BAND' : 'COUNT'})

bar_chart_viz_counts(agg_data_species_event_month, 'EVENT_MONTH', 'SPECIES_NAME', 'SPECIES BANDING EVENTS BY MONTH', True, False)


# <b>Who is observing/banding the birds, and how are the observations occurring?</b>:<br>
# Using the WHO_OBTAINED and HOW_OBTAINED fields to assess the relationship with stacked bar charts. Note that a vast majority of the dataset contains 'unknown' in the WHO_OBTAINED and HOW_OBTAINED fields. These rows will be dropped. Population percentages will be generated for each respective species category to establish an idea of who logged the event and how it occurred.
# 
# <b>The visualizations below some interesting trends to light</b>:
# 1. The 'WHO_OBTAINED' visual shows that some species have a much higher event frequency by bird banders as opposed to finders. 76% of Western Snowy Plover, 36% of Snowy Plover, and 20% of Semipalmated Plover events were reported by bird banders. In contrast, only 12% of Piping Plover and and 8% of Wilson's Plover were reported by bird banders. 100% of Black-bellied Plover events were logged by finder. Additionally, Western Snowy Plover has the highest percentage of reports by State, Provincial, or Federal, with a value of 14%. 
# 2. The 'HOW_OBTAINED' visual shows a broad spread of HOW_OBTAINED for all species except Black-bellied Plover, which is 100% 'saw or photographed neck collar, color band, or other marker (not federal band) while bird was free'. This is the most frequent value for all species with the exception of the Western Snowy Plover, for which 'previously banded bird trapped and released during banding operations' was the most frequent value. Snowy Plover had the largest frequency of 'found dead bird', with a value of 23%. Furthermore, the only species with 'shot.' was Snowy Plover with a value of almost 5%. 

# In[11]:


# first step is to subset the dataframe and create crosstabs of who collected the data and where it was collected
who_obtained_df = cleaned_data[["SPECIES_NAME", "WHO_OBTAINED", "ISO_SUBDIVISION"]]
who_obtained_df = who_obtained_df[who_obtained_df['WHO_OBTAINED'] != 'unknown']
who_obtained_crosstab = pd.crosstab(who_obtained_df['WHO_OBTAINED'],
                            who_obtained_df['SPECIES_NAME'], 
                               margins = False)
who_obtained_crosstab


# In[12]:


def bar_chart_pop_perc (df_in, field_x, field_y, field_color, title, uniform_text_size, legend, height):
    
    fig = px.bar(df_in, x=field_x, y=field_y, color=field_color, 
                    barmode='group', text='PERCENTAGE')

    fig.update_layout(title=title,
                 xaxis=dict(title=field_x, tickmode='linear'))
    
    if legend is not False:
        fig.update_layout(legend=legend)
        
    if uniform_text_size is not False:
        fig.update_layout(uniformtext_minsize=uniform_text_size, uniformtext_mode='show')
        
    if height is not False:
        fig.update_layout(height=height)
        
        
    fig.show()


# In[13]:


# next, calculate population percentages by WHO_OBTAINED and SPECIES_NAME and visualize results
who_obtained_perc = pd.crosstab(who_obtained_df.WHO_OBTAINED,
                                           who_obtained_df.SPECIES_NAME).apply(
    lambda r:r/r.sum(),axis=0)

who_obtained_perc = who_obtained_perc.stack().reset_index().rename(columns={0: 'PERCENTAGE'})
who_obtained_perc['PERCENTAGE'] = (who_obtained_perc['PERCENTAGE'] * 100).round(3) 

bar_chart_pop_perc(who_obtained_perc, 'SPECIES_NAME', 'PERCENTAGE', 'WHO_OBTAINED', 'WHO OBTAINED BAND DATA BY SPECIES', False, False, 600)


# In[14]:


# repeating the same steps for HOW_OBTAINED to visualize how the data were obtained
# first step is to subset the dataframe and create crosstabs of who collected the data and where it was collected
how_obtained_df = cleaned_data[["SPECIES_NAME", "HOW_OBTAINED", "ISO_SUBDIVISION"]]
how_obtained_df = how_obtained_df[how_obtained_df['HOW_OBTAINED'] != 'unknown']
how_obtained_crosstab = pd.crosstab(how_obtained_df['HOW_OBTAINED'],
                            how_obtained_df['SPECIES_NAME'], 
                               margins = False)
how_obtained_crosstab


# In[15]:


# next, calculate population percentages by HOW_OBTAINED and SPECIES_NAME and visualize results
how_obtained_perc = pd.crosstab(how_obtained_df.HOW_OBTAINED,
                                           how_obtained_df.SPECIES_NAME).apply(
    lambda r:r/r.sum(),axis=0)

how_obtained_perc = how_obtained_perc.stack().reset_index().rename(columns={0: 'PERCENTAGE'})
how_obtained_perc['PERCENTAGE'] = (how_obtained_perc['PERCENTAGE'] * 100).round(3)

legend=dict(
    yanchor="bottom",
    y=-0.6,
    xanchor="left",
    x=0.01
)

text_size = 4

bar_chart_pop_perc(how_obtained_perc, 'SPECIES_NAME', 'PERCENTAGE', 'HOW_OBTAINED', 'HOW OBTAINED EVENT DATA BY SPECIES', 
                   text_size, legend, 900)


# <b>What is the status of the bird during the event?</b>:<br>
# 
# In the analysis below, we're looking at whether the bird is a normal wild bird, is in rehabilitation, is sick/exhausted/injured, transported, hand-reared/gamed/hacked, or held for experimental purposes. Note that a large number of entries were 'unknown' for the BIRD_STATUs_DESCRIPTION field, so these rows will be dropped for the purpose of this analysis.
# 
# Population percentages will be calculated via crosstabulation, then T-Test statistical analysis will be executed on emerging trends. In the data below, Piping Plover shows a higher percentage of 'hand-reared/gamed/hacked' as BIRD_STATUS_DESCRIPTION. T-Test will be used to confirm whether the difference is significant compared to the other species.

# In[16]:


species_and_status = cleaned_data.groupby(['SPECIES_NAME', 'BIRD_STATUS_DESCRIPTION'], 
                                             as_index=False)['BAND'].count().rename(columns={'BAND' : 'COUNT'})

species_and_status_counts = species_and_status[species_and_status.BIRD_STATUS_DESCRIPTION != 'unknown']

legend=dict(
    yanchor="bottom",
    y=-0.8,
    xanchor="left",
    x=0.01)
    
bar_chart_viz_counts(species_and_status_counts, 'SPECIES_NAME', 'BIRD_STATUS_DESCRIPTION', 'SPECIES VS DESCRIPTION DURING EVENT', False, legend)

species_and_status_counts


# In[17]:


grouped_spec_stat_sums = species_and_status_counts.groupby(['SPECIES_NAME', 
                                                            'BIRD_STATUS_DESCRIPTION'])['COUNT'].agg('sum')
grouped_spec_stat_percs = grouped_spec_stat_sums.groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).reset_index().rename(columns={'COUNT' : 'PERCENTAGE'})

# creating subset of dataframe to assess the relationship between species and bird description
species_status = cleaned_data[["BAND", "SPECIES_NAME", "BIRD_STATUS_DESCRIPTION"]]
species_status.describe()


# In[18]:


# creating a crosstab to help perform population proportions
species_status_crosstab = pd.crosstab(species_status['BIRD_STATUS_DESCRIPTION'],
                            species_status['SPECIES_NAME'], 
                               margins = False)

# converting counts to percentages
# point of interest - appears that piping plover has higher percentage of 'hand-reared, game-farm or hacked bird.'
# let us test this hypothesis to determine significance
species_status_crosstab_perc = pd.crosstab(species_status.BIRD_STATUS_DESCRIPTION,
                                           species_status.SPECIES_NAME).apply(lambda r:r/r.sum(),axis=0)
species_status_crosstab_perc


# In[19]:


# calculating standard error -- pt 1
proportions = species_status.groupby("SPECIES_NAME")["BIRD_STATUS_DESCRIPTION"].agg(
    [lambda z: np.mean(z=="hand-reared, game-farm or hacked bird."), "size"])
proportions.columns = ['prop_hr_gf_hb','total_counts']

# calculating standard error -- pt 2
total_proportion_hr_gf_hb = (
    species_status.BIRD_STATUS_DESCRIPTION == "hand-reared, game-farm or hacked bird.").mean()

variance = total_proportion_hr_gf_hb * (1 - total_proportion_hr_gf_hb)
standard_error = np.sqrt(variance * (1 / proportions.total_counts['black-bellied plover'] +
                                     1 / proportions.total_counts['piping plover'] +
                                    1 / proportions.total_counts['semipalmated plover'] +
                                    1 / proportions.total_counts['snowy plover'] +
                                    1 / proportions.total_counts['western snowy plover']))
print("Sample Standard Error",standard_error)


# In[20]:


# Calculate the test statistic 
best_estimate = (proportions.prop_hr_gf_hb['piping plover'] - (proportions.prop_hr_gf_hb['black-bellied plover'] +
                                                              proportions.prop_hr_gf_hb['semipalmated plover'] +
                                                              proportions.prop_hr_gf_hb['snowy plover'] +
                                                              proportions.prop_hr_gf_hb['western snowy plover']))
print("The best estimate is",best_estimate)
hypothesized_estimate = 0
test_stat = (best_estimate-hypothesized_estimate) / standard_error
print("Computed Test Statistic is",test_stat) # approximately 2 standard errors above the hypothesized estimate


# In[21]:


# computing p-value
import scipy.stats.distributions as dist

pvalue = 2*dist.norm.cdf(-np.abs(test_stat)) # Multiplied by two indicates a two tailed testing.
print("Computed P-value is", pvalue) # p-value is 0.04 


# <b>Result</b>: With a P-Value of 0.04, the hypothesis is validated that Piping Plovers are significantly more likely to be "hand-reared, game-farm or hacked bird." 

# In[22]:


species_and_event_type = cleaned_data.groupby(['SPECIES_NAME', 'EVENT_TYPE_DESCRIPTION'], 
                                             as_index=False)['BAND'].count().rename(columns={'BAND' : 'COUNT'})

species_and_event_counts = species_and_event_type[species_and_event_type.EVENT_TYPE_DESCRIPTION != 'unknown']

bar_chart_viz_counts(species_and_event_counts, 'SPECIES_NAME', 'EVENT_TYPE_DESCRIPTION', 'EVENT TYPE BY SPECIES', True, False)
species_and_event_counts


# In[23]:


species_and_age = cleaned_data.groupby(['SPECIES_NAME', 'AGE'], 
                                             as_index=False)['BAND'].count().rename(columns={'BAND' : 'COUNT'})

species_and_age_counts = species_and_event_type[species_and_age.AGE != 'unknown']

bar_chart_viz_counts(species_and_age, 'SPECIES_NAME', 'AGE', 'AGES OF SPECIES', False, False)
species_and_age


# In[24]:


# this field is nearly identical to the 'EVENT_TYPE' field with the exception of the 'recapture DB' value. 
record_source_vs_species = cleaned_data.groupby(['SPECIES_NAME', 'RECORD_SOURCE_DESCRIPTION'], 
                                             as_index=False)['BAND'].count().rename(columns={'BAND' : 'COUNT'})

record_source_vs_species = record_source_vs_species[record_source_vs_species.RECORD_SOURCE_DESCRIPTION != 'unknown']

bar_chart_viz_counts(record_source_vs_species, 'SPECIES_NAME', 'RECORD_SOURCE_DESCRIPTION', 'COUNT', False, False)
record_source_vs_species


# In[26]:


fig = px.scatter(record_source_vs_species, x='RECORD_SOURCE_DESCRIPTION', y='SPECIES_NAME', color='SPECIES_NAME', secondary_y=True)
fig.show()


# In[ ]:




