
import numpy as np
import pandas as pd

# read in the json files
portfolio = pd.read_json('data/portfolio.json', orient='records', lines=True)
transcript = pd.read_json('data/transcript.json', orient='records', lines=True)

# preprocessing
# portfolio
for c in ['web', 'mobile', 'social']:
    portfolio[c] = [1 if c in r else 0 for r in portfolio['channels']]
portfolio.drop('channels', axis=1, inplace=True)
portfolio.rename(columns={'id': 'offer_id'}, inplace=True)
portfolio.loc[portfolio['offer_type']=='informational', 'reward']=np.nan
portfolio.loc[portfolio['offer_type']=='informational', 'difficulty']=np.nan
# transcript
transcript['value'] = transcript['value'].apply(lambda x: list(x.values())[0])
transcript.rename(columns={'value': 'offer_id'}, inplace=True)
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
# spending
def offer_code(person, time):
    x = list(received[(received['person']==person) &\
                      (received['offer_start']<=time) &\
                      (received['offer_end']>=time)]['offer_code'])
    if len(x) == 0:
        return np.nan
    elif len(x) == 1:
        return x[0]
    else:
        return x
spending = transcript[transcript['event']=='transaction'].loc[:]
spending['offer_code'] = spending.apply(lambda x: offer_code(x['person'],\
                                        x['time']), axis=1)
s = spending.apply(lambda x: pd.Series(x['offer_code']), axis=1).stack()\
                   .reset_index(level=1, drop=True)
s.name = 'offer_code'
spending = spending.drop(['offer_code', 'event'], axis=1).join(s)
spending.rename(columns={'offer_id': 'spending', 'time': 'transaction_time'},\
                inplace=True)
spending.to_csv('data/spending.csv', index=False)