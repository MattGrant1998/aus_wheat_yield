import pickle
import pandas as pd
import xarray as xr
import os


def load_yield_model(state):
    """
    Load the yield model for a given state.

    Args:
        state (str): The state for which to load the yield model.
    Returns:
        sklearn.ensemble.RandomForestRegressor: The loaded yield model.
    """
    model_path = f'/g/data/w97/mg5624/ABS_project/yield_model/{state}/RandomForestRegressor_{state}_wheat_yield.pkl'
    with open(model_path, 'rb') as file:
        yield_model = pickle.load(file)
    return yield_model


def load_predictors_data(state, required_cols=None):
    """
    Load the predictors data for a given state.

    Args:
        state (str): The state for which to load the predictors data.
    Returns:
        pandas.DataFrame: The predictors data for the specified state.
    """
    predictors_path = f'/g/data/w97/mg5624/ABS_project/predictors_per_state/{state}_predictors.nc'
    predictors_data = xr.open_dataset(predictors_path)
    predictors_df = predictors_data.to_dataframe().reset_index()
    print(predictors_df.columns)
    if required_cols is not None:
        predictors_df = predictors_df.dropna(subset=required_cols)
    else:
        predictors_df = predictors_df.dropna()
    
    return predictors_df


def check_year_coverage(raw_df, filtered_df, predictors, state):
    """
    Check whether any years were fully or partially dropped by dropna,
    and report which predictor columns are responsible.

    Args:
        raw_df (pandas.DataFrame): Predictors data before dropna.
        filtered_df (pandas.DataFrame): Predictors data after dropna.
        predictors (list): The predictor columns used by the model.
        state (str): State being checked (for reporting).
    Returns:
        set: Years that are completely missing after filtering.
    """
    raw_years = set(raw_df['time'].unique())
    filtered_years = set(filtered_df['time'].unique())
    missing_years = raw_years - filtered_years

    if missing_years:
        print(f"  [WARNING] {state}: {len(missing_years)} year(s) completely dropped: {sorted(missing_years)}")

        # For each missing year, report which predictor(s) had NaNs causing the drop
        for year in sorted(missing_years):
            year_data = raw_df[raw_df['time'] == year]
            nan_cols = [c for c in predictors if year_data[c].isna().any()]
            nan_fraction = {c: round(year_data[c].isna().mean(), 3) for c in nan_cols}
            print(f"    Year {year}: NaNs found in {nan_cols} (fraction NaN per column: {nan_fraction})")

    # Also check for years with partial (not total) coverage loss, which is
    # less obviously wrong but can still bias annual averages
    raw_counts = raw_df.groupby('time').size()
    filtered_counts = filtered_df.groupby('time').size()
    partial_loss = {}
    for year in filtered_counts.index:
        if year in raw_counts.index and filtered_counts[year] < raw_counts[year]:
            partial_loss[year] = (filtered_counts[year], raw_counts[year])

    if partial_loss:
        print(f"  [INFO] {state}: {len(partial_loss)} year(s) with partial row loss (rows kept/total):")
        for year, (kept, total) in sorted(partial_loss.items()):
            pct = round(100 * kept / total, 1)
            print(f"    Year {year}: {kept}/{total} rows kept ({pct}%)")

    return missing_years


def predict_yield(state, years=None):
    """
    Predict the yield for a given state using the loaded yield model and predictors data.

    Args:
        state (str): The state for which to predict the yield.
        years (list, optional): List of [start_year, end_year] to filter the predictors data. If None, use all available years.
    Returns:
        pandas.DataFrame: A DataFrame containing the predicted yields along with the corresponding year and state.
    """
    yield_model = load_yield_model(state)
    predictors = yield_model.feature_names_in_.tolist()  # Get the feature names used in the model

    # Load raw (unfiltered) predictors to compare against the filtered version
    predictors_path = f'/g/data/w97/mg5624/ABS_project/predictors_per_state/{state}_predictors.nc'
    raw_df = xr.open_dataset(predictors_path).to_dataframe().reset_index()

    predictors_df = load_predictors_data(state, required_cols=predictors)
    if years is not None:
        years_list = list(range(years[0], years[1] + 1))
        predictors_df = predictors_df[predictors_df['time'].isin(years_list)]
        raw_df = raw_df[raw_df['time'].isin(years_list)]
    missing_years = check_year_coverage(raw_df, predictors_df, predictors, state)
    if missing_years:
        # Fail loudly rather than silently producing incomplete output.
        # Comment this out if partial coverage is acceptable and you just
        # want the warning printed.
        raise ValueError(
            f"{state}: predictions cannot be generated for {len(missing_years)} year(s) "
            f"due to missing predictor data: {sorted(missing_years)}"
        )
    
    # Ensure the predictors are in the same order as the model expects
    predictors_ordered = predictors_df[predictors]

    # Predict yields using the loaded model
    predicted_yields = yield_model.predict(predictors_ordered)

    # Create a DataFrame to hold the results, pulling identifiers straight
    # from the original (unfiltered) predictors_df to guarantee alignment
    results_df = pd.DataFrame({
        'year': predictors_df['time'].values,
        'lat': predictors_df['lat'].values,
        'lon': predictors_df['lon'].values,
        'state': state,
        'predicted_yield': predicted_yields
    })

    return results_df


def main():
    states = [
        # 'NSW', 
        # 'VIC', 
        # 'QLD', 
        # 'SA', 
        'WA',
    ]
    years = [1950, 2021]  # Specify the range of years for which to predict yields
    for state in states:
        print(f"Predicting yield for state: {state}")
        state_results = predict_yield(state, years)
    
        pathout = '/g/data/w97/mg5624/ABS_project/yield_model_analytics/modelled_yield/past_yield/'
        if not os.path.exists(pathout):
            os.makedirs(pathout)
        
        fileout = f'{state}_modelled_yield.csv'
        state_results.to_csv(pathout + fileout, index=False)
        print(f"Predicted {state} yields saved to: {pathout + fileout}")


if __name__ == "__main__":
    main()
