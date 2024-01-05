from datetime import datetime
import pandas as pd
import numpy as np
import os

INPUT_FOLDER = os.path.join('data', 'input')
OUTPUT_FOLDER = os.path.join('..', 'd3')
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
# exp_df.drop(columns=['year'], inplace=True)
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
exp_df['deaths'] = exp_df['mdeaths'] + exp_df['hdeaths']
exp_df.drop(columns=['mdeaths', 'hdeaths'], inplace=True)

# Let's look just at Everest for a second
ev_df = exp_df.query('peakid == "EVER"')

# Group by year and season
ev_df = ev_df.groupby(by=['year', 'season'])['deaths'].sum().reset_index()
ev_df.sort_values(by=['year', 'season'], ascending=True, inplace=True, ignore_index=True)
ev_df['ys'] = ev_df['year'].astype(str) + '-' + ev_df['season'].astype(str)

# Create another dataframe with all year season combinations
y_range = range(int(ev_df.year.min()), int(ev_df.year.max()))
s_range = range(1, 4)
ys_range = sorted([f"{y}-{s}" for s in s_range for y in y_range])
ys_df = pd.DataFrame({'ys': ys_range})

ev_df = ev_df.groupby(by='ys')['deaths'].sum().reset_index()
ev_df = ys_df.merge(ev_df, how='left', on='ys')
ev_df['deaths'] = ev_df['deaths'].fillna(0)

# Prepare for plotting in D3
ev_df.sort_values(by='ys', ascending=True, inplace=True, ignore_index=True)
ev_df.drop(columns='ys', inplace=True)
ev_df.reset_index(inplace=True)
ev_df.rename(columns={'index': 'idx'}, inplace=True)
ev_df.query('deaths > 0', inplace=True)
ev_df['is_dashed'] = False
ev_df['season'] = 'winter'

# TODO - temporary formatting
ev_df['deaths'] = 2 * (ev_df['deaths'] / ev_df['deaths'].max())

# Save
ev_df.to_csv(os.path.join(OUTPUT_FOLDER, 'ev_df.csv'), index=False)

print('hi')
