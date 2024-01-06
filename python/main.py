from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import os

# ------------------------------------------------------------------------------- #
#                                    Constants                                    #
# ------------------------------------------------------------------------------- #

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

INPUT_FOLDER = os.path.join('data')
OUTPUT_FOLDER = os.path.join('..', 'd3')
EXPED_FILE = 'expeditions.csv'
PKNME_FILE = 'peaks.csv'
NO_PEAKS = 5
A = 0.5
B = 3
DASHED_THICKNESS = 0.5
DEATHRATE_THRSH = 0.3
EXP_COLS = ['expid',
            'peakid',
            'year',
            'season',
            'success1',
            'success2',
            'success3',
            'success4',
            'totmembers',
            'tothired',
            'smtmembers',
            'smthired',
            'mdeaths',
            'hdeaths']
SUCCPROB_THRSH = 0.7


# ------------------------------------------------------------------------------- #
#                                Read & Clean Data                                #
# ------------------------------------------------------------------------------- #

exp_df = pd.read_csv(os.path.join(INPUT_FOLDER, EXPED_FILE), usecols=EXP_COLS)
exp_df['expid'] = exp_df['expid'] + '-' + exp_df['year'].astype(str)  # Expedition ID not always accounting for century

# Keep only rows where year and season are not missing
exp_df.dropna(subset=['year', 'season'], inplace=True, ignore_index=True)

# Count number of summits per expedition (total number of people that summited)
exp_df['no_summits'] = exp_df['smtmembers'] + exp_df['smthired']
exp_df.drop(columns=['smtmembers', 'smthired'], inplace=True)

# Calculate number of members
exp_df['no_members'] = exp_df['totmembers'] + exp_df['tothired']
exp_df.drop(columns=['totmembers', 'tothired'], inplace=True)

# Reconcile member deaths and hired deaths
exp_df['no_deaths'] = exp_df['mdeaths'] + exp_df['hdeaths']
exp_df.drop(columns=['mdeaths', 'hdeaths'], inplace=True)

# Add has_expedition column
exp_df['no_exped'] = 1

# Group by year, season, and peakid and add count for: summit, deaths, expeditions
exp_df = exp_df.groupby(by=['year', 'season', 'peakid'])[['no_summits', 'no_members', 'no_deaths', 'no_exped']]\
    .sum().reset_index()
exp_df.sort_values(by=['year', 'season', 'peakid'], ascending=True, inplace=True, ignore_index=True)

# Remove rows where no_members = 0, since this makes no sense
exp_df.query('no_members > 0', inplace=True)

# Keep only peaks with at least MIN_EXPED expeditions over all year.seasons (this leaves us with 10 peaks)
key_exp = exp_df['peakid'].value_counts().reset_index()
key_exp.sort_values(by='count', ascending=False, inplace=True, ignore_index=True)
key_exp = key_exp.iloc[:NO_PEAKS, :]
key_exp.drop(columns=['count'], inplace=True)
exp_df = key_exp.merge(exp_df, how='left', on='peakid')


# ------------------------------------------------------------------------------- #
#                                  For Plotting                                   #
# ------------------------------------------------------------------------------- #

# Create all year, season, peakid combinations in the min/max range (used to do to set index appropriately)
y_range = range(int(exp_df.year.min()), int(exp_df.year.max()) + 1)  # Want range over all peaks/expeditions
s_range = range(1, 5)
p_list = list(exp_df.peakid.unique())
ys_range = sorted([(y, s, p) for s in s_range for y in y_range for p in p_list])
ysp_df = pd.DataFrame(ys_range, columns=['year', 'season', 'peakid'])

# Add combinations to exp_df (again, used to do to set index appropriately)
plt_df = ysp_df.merge(exp_df, how='left', on=['year', 'season', 'peakid'])
plt_df['no_summits'] = plt_df['no_summits'].fillna(0)
plt_df['no_members'] = plt_df['no_members'].fillna(0)
plt_df['no_deaths'] = plt_df['no_deaths'].fillna(0)
plt_df['no_exped'] = plt_df['no_exped'].fillna(0)
plt_df['is_good_seas'] = plt_df['no_exped'].gt(0) & plt_df['no_deaths'].eq(0)

# Add high_deathrate flag
plt_df['death_prob'] = np.where(plt_df['no_members'].gt(0), plt_df['no_deaths'] / plt_df['no_members'], 0)
plt_df['high_deathrate'] = plt_df['death_prob'].gt(DEATHRATE_THRSH)
plt_df.drop(columns=['death_prob'], inplace=True)

# Add success prob flag
plt_df['succ_prob'] = np.where(plt_df['no_members'].gt(0), plt_df['no_summits'] / plt_df['no_members'], 0)
plt_df['high_succprob'] = plt_df['succ_prob'].gt(SUCCPROB_THRSH)
plt_df.drop(columns=['succ_prob'], inplace=True)

# Drop unnecessary columns
plt_df.drop(columns=['no_summits', 'no_members'], inplace=True)

# Now set index for each peakid
plt_df.sort_values(by=['peakid', 'year', 'season'], ascending=True, inplace=True, ignore_index=True)
plt_df['idx'] = plt_df.groupby(['peakid']).cumcount()


# ------------------------------------------------------------------------------- #
#                               Save Data For Plot                                #
# ------------------------------------------------------------------------------- #

# Drop unnecessary columns
plt_df = plt_df.drop(columns=['year', 'season'])

# Filter to season/year/peak combos with at least one expedition (no square will be drawn for other combos)
plt_df = plt_df.query('no_exped > 0').copy()

# Add is_dashed column
plt_df['is_dashed'] = plt_df['is_good_seas']

# Scale entire death column (over all records that remain)
non_zero_values = plt_df['no_deaths'][plt_df['no_deaths'] > 0]
log_transformed_values = non_zero_values.apply(np.log)
normalized_non_zero_values =\
    (log_transformed_values - log_transformed_values.min()) / (log_transformed_values.max() - log_transformed_values.min()) * (B - A) + A
scaled_series = plt_df['no_deaths'].copy()
scaled_series[plt_df['no_deaths'] > 0] = normalized_non_zero_values
plt_df['no_deaths'] = scaled_series

# Set deaths to 1 when expeditions but no deaths
plt_df.loc[plt_df['is_good_seas'], 'no_deaths'] = DASHED_THICKNESS

# Keep only cases where death > 0 (didn't do this before since some cases where deaths = 0 were replaced with 1)
plt_df.query('no_deaths > 0', inplace=True)

# Add peak names and save peakid to peak names
pknme_df = pd.read_csv(os.path.join(INPUT_FOLDER, PKNME_FILE), usecols=['peakid', 'pkname'])
plt_df = plt_df.merge(pknme_df, how='left', on='peakid')
pknme_df = plt_df[['peakid', 'pkname']].drop_duplicates().sort_values(by=['peakid'], ascending=True, ignore_index=True)
pknme_df.to_csv(os.path.join(OUTPUT_FOLDER, 'pknme_df.csv'), index=False)

for peakid in sorted(list(plt_df.peakid.unique())):

    # Pick a peak
    peak_df = plt_df.query(f'peakid == "{peakid}"')

    # Save
    peak_df.to_csv(os.path.join(OUTPUT_FOLDER, 'plt_df.csv'), index=False)

    print(peakid)
