import pandas as pd
import os

def summarize_ice(df, var_col, y_col="y_hat"):
    """
    Summarize ICE curves by computing mean and percentile bands
    for each unique value of the variable.

    Parameters
    ----------
    df : pd.DataFrame
        Raw ICE dataframe with many rows per variable value.
    var_col : str
        Column name of the ICE variable (e.g., "planting_seas_frost_sum").
    y_col : str
        Column name of the ICE response (default = "y_hat").

    Returns
    -------
    pd.DataFrame
        Summary dataframe with mean and percentile bands.
    """

    percentiles = {
        "5th_percentile": 0.05,
        "15th_percentile": 0.15,
        "25th_percentile": 0.25,
        "35th_percentile": 0.35,
        "65th_percentile": 0.65,
        "75th_percentile": 0.75,
        "85th_percentile": 0.85,
        "95th_percentile": 0.95,
    }

    # Group by the ICE variable value
    grouped = df.groupby(var_col)[y_col]

    # Compute summary stats
    summary = grouped.mean().rename("mean").to_frame()

    for name, q in percentiles.items():
        summary[name] = grouped.quantile(q)

    summary = summary.reset_index()

    return summary


def main(combine_states=False):
    datadir = '/g/data/w97/mg5624/ABS_project/'
    pdp_datadir = f'{datadir}yield_model_analytics/partial_dependence/'

    season_vars = {
        "planting": [
            "frost_sum",
            "180_day_precip_drought_intensity_mean",
            "180_day_precip_drought_sum"
        ],
        "mid": [
            "frost_sum",
            "180_day_precip_drought_intensity_mean",
            "180_day_precip_drought_sum",
            "heatwave_intensity_mean",
            "heatwave_sum"
        ],
        "harvest": [
            "180_day_precip_drought_intensity_mean",
            "180_day_precip_drought_sum",
            "heatwave_intensity_mean",
            "heatwave_sum"
        ],
        "growing": [
            "frost_sum",
            "180_day_precip_drought_intensity_mean",
            "180_day_precip_drought_sum",
            "heatwave_intensity_mean",
            "heatwave_sum"
        ]
    }

    # States and variables (as before)
    states = [
        'NSW',
        'QLD',
        'VIC',
        'SA',
        'WA'
    ]

    for season in season_vars.keys():
        vars_to_plot = season_vars[season]
        print(f"\n=== Season: {season.upper()} ===")

        for state in states:
            if len(vars_to_plot) == 1:
                axes = [axes]  # ensure iterable

            for var in vars_to_plot:
                filename = f'{season}_seas_{var}_ICE_RFRegressor_wheat_{state}_extremes_only_{season}_season_only.csv'

                filepath = f"{pdp_datadir}/{season}_season/single_drought_180/{filename}"

                ice_df = pd.read_csv(filepath)
                var_label = f"{season}_seas_{var}"
                summary_df = summarize_ice(ice_df, var_col=var_label)

                outfile = f'{season}_seas_{var}_PDP_RFRegressor_wheat_{state}_extremes_only_{season}_season_only.csv'
                outpath = f'/g/data/w97/mg5624/ABS_project/yield_model_analytics/partial_dependence/summaries/{season}_season/single_drought_180/'
                if not os.path.exists(outpath):
                    os.makedirs(outpath)
                summary_df.to_csv(outpath + outfile, index=False)
        
        # --- Combined across all states ---
        if combine_states:
            print(f"Combining ICE curves across all states for season: {season}")

            for var in vars_to_plot:
                var_label = f"{season}_seas_{var}"

                combined_list = []

                for state in states:
                    filename = f'{season}_seas_{var}_ICE_RFRegressor_wheat_{state}_extremes_only_{season}_season_only.csv'
                    filepath = f"{pdp_datadir}/{season}_season/single_drought_180/{filename}"

                    if os.path.exists(filepath):
                        df_state = pd.read_csv(filepath)
                        df_state["state"] = state
                        combined_list.append(df_state)

                if len(combined_list) == 0:
                    continue

                combined_df = pd.concat(combined_list, ignore_index=True)
                summary_df = summarize_ice(combined_df, var_col=var_label)

                outfile = f'{season}_seas_{var}_PDP_RFRegressor_wheat_AUS_extremes_only_{season}_season_only.csv'
                outpath = f'{pdp_datadir}/summaries/{season}_season/single_drought_180/'
                if not os.path.exists(outpath):
                    os.makedirs(outpath)

                summary_df.to_csv(outpath + outfile, index=False)
                print(f"Saved combined summary: {outfile}")

if __name__ == "__main__":
    main(combine_states=False)
    