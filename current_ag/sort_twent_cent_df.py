import pandas as pd

twentieth_cent_data_path = '/g/data/w97/mg5624/ABS_project/twentieth_cent_crop/'
twentieth_cent_data_file = twentieth_cent_data_path + 'aus_data.csv'
twent_cent_data = pd.read_csv(twentieth_cent_data_file)

twent_cent_relevant_states = twent_cent_data[twent_cent_data['admin1'].isin(
    ['New South Wales(b)', 'Queensland', 'South Australia', 'Victoria', 'Western Australia']
)]

twent_cent_relevant_states = twent_cent_relevant_states.drop(columns=['admin0', 'crop', 'notes'])

# Set admin1 as index before transposing so state names become column headers
twent_cent_relevant_states = twent_cent_relevant_states.set_index('admin1')

# Now transpose — years become rows, states become columns
twent_cent_relevant_states_trans = twent_cent_relevant_states.T
twent_cent_relevant_states_trans.index.name = 'Year'
twent_cent_relevant_states_trans.columns.name = None 

# Rename to abbreviations
twent_cent_relevant_states_trans = twent_cent_relevant_states_trans.rename(columns={
    'New South Wales(b)': 'NSW',
    'Queensland': 'QLD',
    'South Australia': 'SA',
    'Victoria': 'VIC',
    'Western Australia': 'WA'
})

twent_cent_relevant_states_trans = twent_cent_relevant_states_trans.reset_index()
twent_cent_relevant_states_trans.to_csv(twentieth_cent_data_path + 'aus_data_sorted.csv')