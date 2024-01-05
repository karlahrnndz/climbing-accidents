from datetime import datetime
import pandas as pd
import numpy as np
import os

INPUT_FOLDER = os.path.join('data', 'input')
EXPED_FILE = 'expeditions.csv'
EXP_COLS = ['expid',
            'peakid',
            'year',
            'season',
            'success1',
            'success2',
            'success3',
            'success4',
            'disputed',
            'claimed',
            'bcdate',
            'smtdate',
            'termdate',
            'termreason',
            'termnote',
            'highpoint',
            'mdeaths',
            'hdeaths',
            'accidents']
DATE_COLS = ['bcdate', 'smtdate', 'termdate']

exp_df = pd.read_csv(os.path.join(INPUT_FOLDER, EXPED_FILE), usecols=EXP_COLS)
exp_df['expid'] = exp_df['expid'] + '-' + exp_df['year'].astype(str)  # Expedition ID not always accounting for century
exp_df['has_summit'] = exp_df[['success1', 'success2', 'success3', 'success4']].apply(lambda x: x.sum() > 0, axis=1)
exp_df.drop(columns=['success1', 'success2', 'success3', 'success4'], inplace=True)
exp_df[DATE_COLS] = exp_df[DATE_COLS].apply(pd.to_datetime, errors='coerce')

# Choosing smtdate > bcdate > termdate to represent expedition date
exp_df['date'] = np.where(exp_df['smtdate'].isna(), exp_df['bcdate'], exp_df['smtdate'])
exp_df['date'] = np.where(exp_df['date'].isna(), exp_df['termdate'], exp_df['date'])
exp_df.drop(columns=['bcdate', 'smtdate', 'termdate'], inplace=True)
exp_df['date'] = np.where((exp_df['date'].dt.year - exp_df['year']).gt(1),
                          exp_df['date'] - pd.DateOffset(years=100),
                          exp_df['date'])  # Pandas might be rounding up the century when reading in dates
exp_df.drop(columns=['year'], inplace=True)
exp_df.rename(columns={'smtdate': 'hpdate'}, inplace=True)

# Keep only rows where date is not missing
exp_df = exp_df.loc[~exp_df['date'].isna(), :]
exp_df.reset_index(drop=True, inplace=True)

# Remove cases where the summit is claimed or disputed
exp_df.query('claimed == False', inplace=True)
exp_df.query('disputed == False', inplace=True)
exp_df.reset_index(inplace=True, drop=True)
exp_df.drop(columns=['claimed', 'disputed'], inplace=True)

# Reconcile member deaths and hired deaths
exp_df['tot_deaths'] = exp_df['mdeaths'] + exp_df['hdeaths']
exp_df.drop(columns=['mdeaths', 'hdeaths'], inplace=True)

# Let's look just at Everest for a second
ev_df = exp_df.query('peakid == "EVER"')

# Group by date
ev_df = ev_df.groupby(by=['date'])['tot_deaths'].sum().reset_index()
ev_df.sort_values(by='date', ascending=True, inplace=True, ignore_index=True)

# Create another dataframe with all days from ev_df.date.min() to ev_df.date.max() (by day)
date_range = pd.date_range(start=ev_df.date.min(), end=ev_df.date.max())
date_range_df = pd.DataFrame({'date': date_range})
ev_df = date_range_df.merge(ev_df, how='left', on='date')
ev_df['tot_deaths'] = ev_df['tot_deaths'].fillna(0)

# Group by month because there are too many days
ev_df['ym'] = ev_df['date'].dt.strftime('%Y-%m')
ev_df = ev_df.groupby(by='ym')['tot_deaths'].sum().reset_index()

print('hi')
