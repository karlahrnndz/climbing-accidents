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
DEATHRATE_THRSH = 0.1
SUCCPROB_THRSH = 0.7
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


# ------------------------------------------------------------------------------- #
#                                Read & Clean Data                                #
# ------------------------------------------------------------------------------- #

exp_df = pd.read_csv(os.path.join(INPUT_FOLDER, EXPED_FILE), usecols=EXP_COLS)

# Keep only rows where year and season are not missing
exp_df.dropna(subset=['year'], inplace=True, ignore_index=True)

# Update expedition ID not always accounting for century
exp_df['expid'] = exp_df['expid'] + '-' + exp_df['year'].astype(str)

# Count number of summits
exp_df['no_summits'] = exp_df['smtmembers'] + exp_df['smthired']
exp_df.drop(columns=['smtmembers', 'smthired'], inplace=True)

# Count number of members
exp_df['no_members'] = exp_df['totmembers'] + exp_df['tothired']
exp_df.drop(columns=['totmembers', 'tothired'], inplace=True)

# Count number of deaths
exp_df['no_deaths'] = exp_df['mdeaths'] + exp_df['hdeaths']
exp_df.drop(columns=['mdeaths', 'hdeaths'], inplace=True)

# Add has_expedition column (all = 1 for now, will be relevant later)
exp_df['no_exped'] = 1

# Group by year and peakid and add count for: no_summits, no_members, no_deaths, and no_exped
exp_df = exp_df.groupby(by=['year', 'peakid'])[['no_summits', 'no_members', 'no_deaths', 'no_exped']]\
    .sum().reset_index()
exp_df.sort_values(by=['year', 'peakid'], ascending=True, inplace=True, ignore_index=True)

# Remove rows where no_members = 0, since no_members ==0 makes no sense (must be error in data)
exp_df.query('no_members > 0', inplace=True)

# Keep only NO_PEAKS with the highest number of expeditions
key_exp = exp_df.groupby(by='peakid')[['no_exped']].sum().reset_index()
key_exp = key_exp.sort_values(by='no_exped', ascending=False, ignore_index=False)
key_exp = key_exp.iloc[:NO_PEAKS, :]
key_exp.drop(columns=['no_exped'], inplace=True)
exp_df = key_exp.merge(exp_df, how='left', on='peakid')


# ------------------------------------------------------------------------------- #
#                                  For Plotting                                   #
# ------------------------------------------------------------------------------- #

# Create all year, season, peakid combinations in the min/max range (used to do to set index appropriately)
y_range = range(int(exp_df.year.min()), int(exp_df.year.max()) + 1)  # Want range over all peaks/expeditions
p_list = list(exp_df.peakid.unique())
ys_range = sorted([(y, p) for y in y_range for p in p_list])
ys_df = pd.DataFrame(ys_range, columns=['year', 'peakid'])

# Add combinations to exp_df (again, used to do to set index appropriately)
plt_df = ys_df.merge(exp_df, how='left', on=['year', 'peakid'])
plt_df['no_summits'] = plt_df['no_summits'].fillna(0)
plt_df['no_members'] = plt_df['no_members'].fillna(0)
plt_df['no_deaths'] = plt_df['no_deaths'].fillna(0)
plt_df['no_exped'] = plt_df['no_exped'].fillna(0)
plt_df['is_good_seas'] = plt_df['no_exped'].gt(0) & plt_df['no_deaths'].eq(0)

# Add high_deathrate flag
plt_df['deathrate'] = np.where(plt_df['no_members'].gt(0), plt_df['no_deaths'] / plt_df['no_members'], 0)
plt_df['high_deathrate'] = plt_df['deathrate'].gt(DEATHRATE_THRSH)
plt_df.drop(columns=['deathrate'], inplace=True)

# Add success prob flag
plt_df['succrate'] = np.where(plt_df['no_members'].gt(0), plt_df['no_summits'] / plt_df['no_members'], 0)
plt_df['high_succrate'] = plt_df['succrate'].gt(SUCCPROB_THRSH)
plt_df.drop(columns=['succrate'], inplace=True)

# Drop unnecessary columns
plt_df.drop(columns=['no_summits', 'no_members'], inplace=True)

# Now set index for each peakid over years (by sorting and grouping)
plt_df.sort_values(by=['peakid', 'year'], ascending=True, inplace=True, ignore_index=True)
plt_df['idx'] = plt_df.groupby(['peakid']).cumcount()


# ------------------------------------------------------------------------------- #
#                               Save Data For Plot                                #
# ------------------------------------------------------------------------------- #

# Drop unnecessary columns
# plt_df = plt_df.drop(columns=['year'])

# Filter to year/peak combos with at least one expedition (no square will be drawn for other combos anyway)
plt_df = plt_df.query('no_exped > 0').copy()

# Add is_dashed column
plt_df['is_dashed'] = plt_df['is_good_seas']

# Log-transform and normalize entire no_deaths column (over all records that remain)
non_zero_values = plt_df['no_deaths'][plt_df['no_deaths'] > 0]
log_transformed_values = non_zero_values.apply(np.log)  # Log-transform
normalized_non_zero_values = (log_transformed_values - log_transformed_values.min())\
                             / (log_transformed_values.max() - log_transformed_values.min()) * (B - A) + A  # Normalize
plt_df.loc[plt_df['no_deaths'].gt(0), 'no_deaths'] = normalized_non_zero_values

# Set deaths to DASHED_THICKNESS when there are expeditions but no deaths
plt_df.loc[plt_df['is_good_seas'], 'no_deaths'] = DASHED_THICKNESS

# Keep cases where death > 0 (before this some cases where deaths == 0 needed to be replaced with DASHED_THICKNESS)
plt_df.query('no_deaths > 0', inplace=True)

# Create dataframe with peak names
pknme_df = pd.read_csv(os.path.join(INPUT_FOLDER, PKNME_FILE), usecols=['peakid', 'pkname'])
pknme_df.drop_duplicates(inplace=True, ignore_index=True)  # Just in case

# Add peak names to plt_df and save named of used peaks
plt_df = plt_df.merge(pknme_df, how='left', on='peakid')
plt_df[['peakid', 'pkname']].drop_duplicates().to_csv(os.path.join(OUTPUT_FOLDER, 'pknme_df.csv'), index=False)

for peakid in sorted(list(plt_df['peakid'].unique())):

    # Pick a peak
    peak_df = plt_df.query(f'peakid == "{peakid}"')

    # Save data for D3 visualization
    peak_df.to_csv(os.path.join(OUTPUT_FOLDER, 'plt_df.csv'), index=False)

    print(peakid)
    print(peakid)
