import pandas as pd


datadir = '/g/data/w97/mg5624/ABS_project/yield_model_analytics/modelled_yield/'

def load_modelled_yield(state, years=None):
    """
    Load the modelled yield data for a given state.

    Args:
        state (str): The state for which to load the modelled yield data.
    Returns:
        pandas.DataFrame: The modelled yield data for the specified state.
    """
    if state == 'AUS':
        all_states = ['NSW', 'VIC', 'QLD', 'SA', 'WA']
        modelled_yield_data = pd.concat([load_modelled_yield(s, years) for s in all_states], ignore_index=True)
    else:
        if years is None:
            modelled_yield_path = f'{datadir}past_yield/{state}_modelled_yield.csv'
        else:
            modelled_yield_path = f'{datadir}past_yield/{state}_modelled_yield_{years[0]}-{years[1]}.csv'
        modelled_yield_data = pd.read_csv(modelled_yield_path)
    return modelled_yield_data


def compute_annual_avg_yield(modelled_yield_data):
    """
    Compute the annual average yield from the modelled yield data.

    Args:
        modelled_yield_data (pandas.DataFrame): The modelled yield data.
    Returns:
        pandas.DataFrame: A DataFrame containing the annual average yield.
    """
    print(modelled_yield_data.head())
    annual_avg_yield_df = modelled_yield_data.groupby('year')['predicted_yield'].mean().reset_index()
    return annual_avg_yield_df


def main():
    states = [
        'AUS', 
        'NSW', 
        'VIC', 
        'QLD', 
        'SA', 
        'WA'
    ]
    
    all_results = []
    years = [2022, 2025]
    for state in states:
        print(f"Processing annual average yield for state: {state}")
        modelled_yield_data = load_modelled_yield(state, years=years)
        annual_avg_yield_df = compute_annual_avg_yield(modelled_yield_data)
        annual_avg_yield_df['state'] = state
        all_results.append(annual_avg_yield_df)

    combined_df = pd.concat(all_results, ignore_index=True)
    combined_df = combined_df[['state', 'year', 'predicted_yield']]  # reorder columns

    if years is None:
        output_path = f'{datadir}past_yield/all_states_modelled_annual_avg_yield.csv'
    else:
        output_path = f'{datadir}past_yield/all_states_modelled_annual_avg_yield_{years[0]}-{years[1]}.csv'
    combined_df.to_csv(output_path, index=False)
    print(f"Annual average yield for all states saved to: {output_path}")


if __name__ == "__main__":
    main()
