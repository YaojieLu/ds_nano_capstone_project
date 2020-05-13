
import numpy as np
import pandas as pd

# read in files
portfolio = pd.read_json('data/portfolio.json', orient='records', lines=True)
profile = pd.read_json('data/profile.json', orient='records', lines=True)
transcript = pd.read_json('data/transcript.json', orient='records', lines=True)
viewed = pd.read_csv('data/viewed.csv')
completed = pd.read_csv('data/completed.csv')
spending = pd.read_csv('data/spending.csv')

# preprocessing
# transcript
transcript['value'] = transcript['value'].apply(lambda x: list(x.values())[0])
transcript.rename(columns={'value': 'offer_id'}, inplace=True)
# profile
profile['gender'].replace([None], np.nan, inplace=True)
profile['age'].replace(118, np.nan, inplace=True)
profile.set_index('id', inplace=True)
profile['member_year'] = profile['became_member_on'].apply(lambda x:\
                         str(int(x/10000)))
profile.drop('became_member_on', axis=1, inplace=True)
total_spending = transcript[transcript['event']=='transaction']\
                 .groupby('person')['offer_id'].sum()
profile = pd.concat([profile, total_spending], axis=1).reset_index()
profile.rename(columns={'index': 'person', 'offer_id': 'total_spending'},\
               inplace=True)
profile['total_spending'].fillna(0, inplace=True)
# portfolio
for c in ['web', 'mobile', 'social']:
    portfolio[c] = [1 if c in r else 0 for r in portfolio['channels']]
portfolio.drop('channels', axis=1, inplace=True)
portfolio.rename(columns={'id': 'offer_id'}, inplace=True)
portfolio.loc[portfolio['offer_type']=='informational', 'reward']=np.nan
portfolio.loc[portfolio['offer_type']=='informational', 'difficulty']=np.nan
# offers
offers = transcript[transcript['event']=='offer received'].loc[:,\
                   ['time', 'offer_id']]
offers = offers.drop_duplicates()
offers = pd.merge(offers, portfolio, on='offer_id', how='outer')
offers['offer_end'] = offers.apply(lambda x: x['time']+x['duration']*24,\
                      axis=1)
offers = offers.sort_values(['time', 'offer_end', 'offer_id'])\
                            .reset_index().drop('index', axis=1)
offers.reset_index(inplace=True)
offers.rename(columns={'time': 'offer_start', 'index': 'offer_code'},\
              inplace=True)
# received
received = transcript[transcript['event']=='offer received'].loc[:]
received.drop('event', axis=1, inplace=True)
received.rename(columns={'time': 'offer_start'}, inplace=True)
received = pd.merge(received, offers, on=['offer_id', 'offer_start'],\
                    how='outer')
# combine received, reviewed, completed, spending
received_add = pd.DataFrame({'person': list(profile['person']),\
                             'offer_code': [999]*len(profile),\
                             'offer_type': [999]*len(profile)})#999: no_offer
received = received.append(received_add, ignore_index=True)
viewed.dropna(inplace=True)
df = pd.merge(viewed, completed, how='left',\
              on=['person', 'offer_id', 'offer_code'])
df = pd.merge(received, df, how='left', on=['person', 'offer_id', 'offer_code'])
spending['offer_code'].fillna(999, inplace=True)#999: no_offer
spending = spending.groupby(['person', 'offer_code'])['spending'].sum()\
                   .reset_index()
df = pd.merge(df, spending, how='left', on=['person', 'offer_code'])
df['spending'].fillna(0, inplace=True)
df['viewed'] = df['viewed_time'].notnull()
df['completed'] = df['completed_time'].notnull()
df = pd.merge(df, profile, on='person')
df.to_csv('data/df.csv', index=False)
