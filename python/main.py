import matplotlib.pyplot as plt
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

# Keep only rows where year and season are not missing
exp_df.dropna(subset=['year', 'season'], inplace=True, ignore_index=True)
exp_df.reset_index(drop=True, inplace=True)

# Reconcile member deaths and hired deaths
exp_df['deaths'] = exp_df['mdeaths'] + exp_df['hdeaths']
exp_df.drop(columns=['mdeaths', 'hdeaths'], inplace=True)

# Let's look just at Everest for a second
ev_df = exp_df.query('peakid == "ANN1"')

# Group by year and season and count deaths
ev_df = ev_df.groupby(by=['year', 'season'])['deaths'].sum().reset_index()
ev_df.sort_values(by=['year', 'season'], ascending=True, inplace=True, ignore_index=True)

# Mark cases where there are expeditions but no deaths
ev_df['has_exped'] = True

# Add all year season combinations in the min/max range
y_range = range(int(ev_df.year.min()), int(ev_df.year.max()) + 1)
s_range = range(1, 5)
ys_range = sorted([(y, s) for s in s_range for y in y_range])
ys_df = pd.DataFrame(ys_range, columns=['year', 'season'])
ev_df = ys_df.merge(ev_df, how='left', on=['year', 'season'])
ev_df['deaths'] = ev_df['deaths'].fillna(0)
ev_df['has_exped'] = ev_df['has_exped'].fillna(False)

# Prepare for plotting in D3
ev_df['is_dashed'] = ev_df['has_exped'] & ev_df['deaths'].eq(0)
ev_df.sort_values(by=['year', 'season'], ascending=True, inplace=True, ignore_index=True)
ev_df['season'] = np.where(ev_df['is_dashed'], 'fall', 'spring')
ev_df.reset_index(inplace=True)
ev_df.rename(columns={'index': 'idx'}, inplace=True)
ev_df.query('has_exped == True', inplace=True)
ev_df.drop(columns=['year', 'has_exped'], inplace=True)

ev_df['deaths'] = 2.5 * (ev_df['deaths'] / ev_df['deaths'].max())
ev_df.loc[ev_df['is_dashed'], 'deaths'] = 0.3
ev_df.query('deaths > 0', inplace=True)

# Save
ev_df.to_csv(os.path.join(OUTPUT_FOLDER, 'ev_df.csv'), index=False)

print('hi')
