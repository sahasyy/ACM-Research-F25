#the 10 common features

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def safe_timestamp_conversion(series, game_type, time_col):
    """
    Safely convert timestamps to datetime with debugging info.
    """
    print(f"\nDebugging timestamps for {game_type}:")
    print(f"Sample raw values: {series.head().tolist()}")
    print(f"Min value: {series.min()}")
    print(f"Max value: {series.max()}")
    print(f"Data type: {series.dtype}")

    # Game-specific timestamp handling based on your data formats
    if game_type == 'game1':
        # Game 1 uses seconds (10 digits, e.g., 1421157320)
        print("Game 1: Converting timestamps as seconds...")
        converted = pd.to_datetime(series, unit='s')
    elif game_type == 'game2':
        # Game 2 uses milliseconds (13 digits, e.g., 1406267268241)
        print("Game 2: Converting timestamps as milliseconds...")
        converted = pd.to_datetime(series, unit='ms')
    elif game_type == 'game3':
        # Game 3 format not specified, try to auto-detect
        if series.max() > 1e12:  # Likely milliseconds
            print("Game 3: Timestamps appear to be in milliseconds...")
            converted = pd.to_datetime(series, unit='ms')
        elif series.max() > 1e9:  # Likely seconds
            print("Game 3: Timestamps appear to be in seconds...")
            converted = pd.to_datetime(series, unit='s')
        else:
            print("Game 3: Attempting direct datetime conversion...")
            converted = pd.to_datetime(series)
    else:
        # Fallback: auto-detect based on magnitude
        if series.max() > 1e12:
            print("Auto-detect: Converting as milliseconds...")
            converted = pd.to_datetime(series, unit='ms')
        else:
            print("Auto-detect: Converting as seconds...")
            converted = pd.to_datetime(series, unit='s')

    print(f"Converted sample: {converted.head().tolist()}")
    print(f"Date range: {converted.min()} to {converted.max()}")

    return converted


def calculate_common_features(df, game_type, observation_period_days=5,
                                             churn_prediction_days=10, consecutive_threshold_hours=24):
    """
    Calculate features with individual observation periods and churn prediction periods.

    Args:
        df: Input DataFrame with game data
        game_type: 'game1', 'game2', or 'game3'
        observation_period_days: Number of days for observation period (OP)
        churn_prediction_days: Number of days for churn prediction period (CP)
        consecutive_threshold_hours: Threshold for consecutive play detection

    Returns:
        DataFrame with features and churn labels
    """
    # Standardize column names based on game type
    if game_type == 'game1':
        user_col = 'device'
        time_col = 'time'
        score_col = 'score'
    elif game_type == 'game2':
        user_col = 'device.id'
        time_col = 'time'
        score_col = 'score'
    elif game_type == 'game3':
        user_col = 'id'
        time_col = 'time'
        score_col = 'score'
    else:
        raise ValueError("game_type must be 'game1', 'game2', or 'game3'")

    # Make a copy to avoid modifying original
    df_work = df.copy()

    print(f"\nProcessing {game_type} data with individual OP/CP:")
    print(f"Original shape: {df_work.shape}")
    print(f"Observation Period: {observation_period_days} days")
    print(f"Churn Prediction Period: {churn_prediction_days} days")

    # Convert time to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df_work[time_col]):
        try:
            df_work[time_col] = safe_timestamp_conversion(df_work[time_col], game_type, time_col)
        except Exception as e:
            print(f"Error converting timestamps: {e}")
            return pd.DataFrame()

    # Remove rows with NaT values and missing scores
    df_work = df_work.dropna(subset=[time_col, score_col])

    if df_work.empty:
        print("No valid data after cleaning")
        return pd.DataFrame()

    # Sort by user and time
    df_work = df_work.sort_values([user_col, time_col])

    print(f"Data shape after cleaning: {df_work.shape}")

    # Calculate features for each user with individual OP/CP
    user_features = []
    users_processed = 0
    users_with_enough_data = 0

    for user_id, user_data in df_work.groupby(user_col):
        user_data = user_data.sort_values(time_col).reset_index(drop=True)
        users_processed += 1

        if len(user_data) == 0:
            continue

        # Individual timing: t1 = 0 is the user's first play time
        first_play_time = user_data[time_col].iloc[0]

        # Calculate individual observation period end (OP days from first play)
        individual_op_end = first_play_time + timedelta(days=observation_period_days)

        # Calculate individual churn prediction period end (CP days after OP)
        individual_cp_end = individual_op_end + timedelta(days=churn_prediction_days)

        # Filter plays within observation period only
        op_plays = user_data[user_data[time_col] <= individual_op_end].copy()

        # Need at least one play in OP to create features
        if len(op_plays) == 0:
            continue

        users_with_enough_data += 1

        # Calculate features using only observation period data
        scores = op_plays[score_col].values
        play_count = len(scores)

        # Active duration within OP
        last_play_in_op = op_plays[time_col].iloc[-1]
        active_duration = (last_play_in_op - first_play_time).total_seconds() / 3600  # in hours

        # Score-based features
        mean_score = np.mean(scores)
        best_score = np.max(scores)
        worst_score = np.min(scores)
        sd_score = np.std(scores, ddof=1) if play_count > 1 else 0

        # Best score index (normalized by play count in OP)
        best_score_index = np.argmax(scores) / play_count if play_count > 0 else 0

        # Best score minus mean score features
        best_sub_mean = best_score - mean_score
        best_sub_mean_count = best_sub_mean / play_count if play_count > 0 else 0
        best_sub_mean_ratio = best_sub_mean / mean_score if mean_score != 0 else 0

        # Consecutive play ratio within OP
        consecutive_play_ratio = 0
        if play_count > 1:
            time_diffs = op_plays[time_col].diff().dt.total_seconds() / 3600  # in hours
            time_diffs = time_diffs[1:]  # Remove first NaN
            consecutive_count = np.sum(time_diffs < consecutive_threshold_hours)
            consecutive_play_ratio = consecutive_count / (play_count - 1)

        # CHURN DETERMINATION using individual CP
        # Check if user played during churn prediction period
        cp_plays = user_data[
            (user_data[time_col] > individual_op_end) &
            (user_data[time_col] <= individual_cp_end)
            ]

        # User is churned if they had NO plays during CP period
        is_churned = 1 if len(cp_plays) == 0 else 0

        # Create feature record
        feature_record = {
            user_col: user_id,
            'activeDuration': active_duration,
            'bestScore': best_score,
            'bestScoreIndex': best_score_index,
            'bestSubMeanCount': best_sub_mean_count,
            'bestSubMeanRatio': best_sub_mean_ratio,
            'consecutivePlayRatio': consecutive_play_ratio,
            'meanScore': mean_score,
            'playCount': play_count,
            'sdScore': sd_score,
            'worstScore': worst_score,
            'firstPlayTime': first_play_time,
            'lastPlayTime': last_play_in_op,
            'individualOPEnd': individual_op_end,
            'individualCPEnd': individual_cp_end,
            'playsInCP': len(cp_plays),
            'is_churned': is_churned,
            'observationPeriodDays': observation_period_days,
            'churnPredictionDays': churn_prediction_days
        }

        user_features.append(feature_record)

    print(f"Processed {users_processed} users")
    print(f"Users with enough data: {users_with_enough_data}")

    if not user_features:
        print("No users had sufficient data for feature calculation")
        return pd.DataFrame()

    # Convert to DataFrame
    features_df = pd.DataFrame(user_features)

    # Print churn statistics
    churn_rate = features_df['is_churned'].mean()
    print(f"Churn rate: {churn_rate:.3f} ({features_df['is_churned'].sum()}/{len(features_df)})")

    return features_df


def create_game_features_pipeline(game1_df, game2_plays_df, game3_df=None,
                                  observation_period_days=5, churn_prediction_days=10,
                                  consecutive_threshold_hours=24):
    """
    Create common features for all three games.

    Args:
        game1_df: Game 1 DataFrame
        game2_plays_df: Game 2 play events DataFrame
        game3_df: Game 3 DataFrame (optional)
        observation_period_days: Number of days for observation period
        churn_prediction_days: Number of days for churn prediction period
        consecutive_threshold_hours: Threshold for consecutive play detection
    """
    results = {}

    # Game 1 Features
    print("Calculating Game 1 common features...")
    try:
        game1_features = calculate_common_features(
            game1_df, 'game1', observation_period_days, churn_prediction_days, consecutive_threshold_hours
        )
        results['game1_features'] = game1_features
    except Exception as e:
        print(f"Error processing Game 1: {e}")
        results['game1_features'] = pd.DataFrame()

    # Game 2 Features (only for play events)
    print("Calculating Game 2 common features...")
    try:
        if not game2_plays_df.empty:
            game2_features = calculate_common_features(
                game2_plays_df, 'game2', observation_period_days, churn_prediction_days, consecutive_threshold_hours
            )
        else:
            game2_features = pd.DataFrame()
        results['game2_features'] = game2_features
    except Exception as e:
        print(f"Error processing Game 2: {e}")
        results['game2_features'] = pd.DataFrame()

    # Game 3 Features (if provided)
    if game3_df is not None:
        print("Calculating Game 3 common features...")
        try:
            game3_features = calculate_common_features(
                game3_df, 'game3', observation_period_days, churn_prediction_days, consecutive_threshold_hours
            )
            results['game3_features'] = game3_features
        except Exception as e:
            print(f"Error processing Game 3: {e}")
            results['game3_features'] = pd.DataFrame()

    return results


import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def safe_timestamp_conversion(series, game_type, time_col):
    """
    Safely convert timestamps to datetime with debugging info.
    """
    print(f"\nDebugging timestamps for {game_type}:")
    print(f"Sample raw values: {series.head().tolist()}")
    print(f"Min value: {series.min()}")
    print(f"Max value: {series.max()}")
    print(f"Data type: {series.dtype}")

    # Game-specific timestamp handling based on your data formats
    if game_type == 'game1':
        # Game 1 uses seconds (10 digits, e.g., 1421157320)
        print("Game 1: Converting timestamps as seconds...")
        converted = pd.to_datetime(series, unit='s')
    elif game_type == 'game2':
        # Game 2 uses milliseconds (13 digits, e.g., 1406267268241)
        print("Game 2: Converting timestamps as milliseconds...")
        converted = pd.to_datetime(series, unit='ms')
    elif game_type == 'game3':
        # Game 3 format not specified, try to auto-detect
        if series.max() > 1e12:  # Likely milliseconds
            print("Game 3: Timestamps appear to be in milliseconds...")
            converted = pd.to_datetime(series, unit='ms')
        elif series.max() > 1e9:  # Likely seconds
            print("Game 3: Timestamps appear to be in seconds...")
            converted = pd.to_datetime(series, unit='s')
        else:
            print("Game 3: Attempting direct datetime conversion...")
            converted = pd.to_datetime(series)
    else:
        # Fallback: auto-detect based on magnitude
        if series.max() > 1e12:
            print("Auto-detect: Converting as milliseconds...")
            converted = pd.to_datetime(series, unit='ms')
        else:
            print("Auto-detect: Converting as seconds...")
            converted = pd.to_datetime(series, unit='s')

    print(f"Converted sample: {converted.head().tolist()}")
    print(f"Date range: {converted.min()} to {converted.max()}")

    return converted


def calculate_common_features(df, game_type, observation_period_days=5,
                              churn_prediction_days=10, consecutive_threshold_hours=24):
    """
    Calculate features with individual observation periods and churn prediction periods.

    Args:
        df: Input DataFrame with game data
        game_type: 'game1', 'game2', or 'game3'
        observation_period_days: Number of days for observation period (OP)
        churn_prediction_days: Number of days for churn prediction period (CP)
        consecutive_threshold_hours: Threshold for consecutive play detection

    Returns:
        DataFrame with features and churn labels
    """
    # Standardize column names based on game type
    if game_type == 'game1':
        user_col = 'device'
        time_col = 'time'
        score_col = 'score'
    elif game_type == 'game2':
        user_col = 'device.id'
        time_col = 'time'
        score_col = 'score'
    elif game_type == 'game3':
        user_col = 'id'
        time_col = 'time'
        score_col = 'score'
    else:
        raise ValueError("game_type must be 'game1', 'game2', or 'game3'")

    # Make a copy to avoid modifying original
    df_work = df.copy()

    print(f"\nProcessing {game_type} data with individual OP/CP:")
    print(f"Original shape: {df_work.shape}")
    print(f"Observation Period: {observation_period_days} days")
    print(f"Churn Prediction Period: {churn_prediction_days} days")

    # Convert time to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df_work[time_col]):
        try:
            df_work[time_col] = safe_timestamp_conversion(df_work[time_col], game_type, time_col)
        except Exception as e:
            print(f"Error converting timestamps: {e}")
            return pd.DataFrame()

    # Remove rows with NaT values and missing scores
    df_work = df_work.dropna(subset=[time_col, score_col])

    if df_work.empty:
        print("No valid data after cleaning")
        return pd.DataFrame()

    # Sort by user and time
    df_work = df_work.sort_values([user_col, time_col])

    print(f"Data shape after cleaning: {df_work.shape}")

    # Calculate features for each user with individual OP/CP
    user_features = []
    users_processed = 0
    users_with_enough_data = 0

    for user_id, user_data in df_work.groupby(user_col):
        user_data = user_data.sort_values(time_col).reset_index(drop=True)
        users_processed += 1

        if len(user_data) == 0:
            continue

        # Individual timing: t1 = 0 is the user's first play time
        first_play_time = user_data[time_col].iloc[0]

        # Calculate individual observation period end (OP days from first play)
        individual_op_end = first_play_time + timedelta(days=observation_period_days)

        # Calculate individual churn prediction period end (CP days after OP)
        individual_cp_end = individual_op_end + timedelta(days=churn_prediction_days)

        # Filter plays within observation period only
        op_plays = user_data[user_data[time_col] <= individual_op_end].copy()

        # Need at least one play in OP to create features
        if len(op_plays) == 0:
            continue

        users_with_enough_data += 1

        # Calculate features using only observation period data
        scores = op_plays[score_col].values
        play_count = len(scores)

        # Active duration within OP
        last_play_in_op = op_plays[time_col].iloc[-1]
        active_duration = (last_play_in_op - first_play_time).total_seconds() / 3600  # in hours

        # Score-based features
        mean_score = np.mean(scores)
        best_score = np.max(scores)
        worst_score = np.min(scores)
        sd_score = np.std(scores, ddof=1) if play_count > 1 else 0

        # Best score index (normalized by play count in OP)
        best_score_index = np.argmax(scores) / play_count if play_count > 0 else 0

        # Best score minus mean score features
        best_sub_mean = best_score - mean_score
        best_sub_mean_count = best_sub_mean / play_count if play_count > 0 else 0
        best_sub_mean_ratio = best_sub_mean / mean_score if mean_score != 0 else 0

        # Consecutive play ratio within OP
        consecutive_play_ratio = 0
        if play_count > 1:
            time_diffs = op_plays[time_col].diff().dt.total_seconds() / 3600  # in hours
            time_diffs = time_diffs[1:]  # Remove first NaN
            consecutive_count = np.sum(time_diffs < consecutive_threshold_hours)
            consecutive_play_ratio = consecutive_count / (play_count - 1)

        # CHURN DETERMINATION using individual CP
        # Check if user played during churn prediction period
        cp_plays = user_data[
            (user_data[time_col] > individual_op_end) &
            (user_data[time_col] <= individual_cp_end)
            ]

        # User is churned if they had NO plays during CP period
        is_churned = 1 if len(cp_plays) == 0 else 0

        # Create feature record
        feature_record = {
            user_col: user_id,
            'activeDuration': active_duration,
            'bestScore': best_score,
            'bestScoreIndex': best_score_index,
            'bestSubMeanCount': best_sub_mean_count,
            'bestSubMeanRatio': best_sub_mean_ratio,
            'consecutivePlayRatio': consecutive_play_ratio,
            'meanScore': mean_score,
            'playCount': play_count,
            'sdScore': sd_score,
            'worstScore': worst_score,
            'firstPlayTime': first_play_time,
            'lastPlayTime': last_play_in_op,
            'individualOPEnd': individual_op_end,
            'individualCPEnd': individual_cp_end,
            'playsInCP': len(cp_plays),
            'is_churned': is_churned,
            'observationPeriodDays': observation_period_days,
            'churnPredictionDays': churn_prediction_days
        }

        user_features.append(feature_record)

    print(f"Processed {users_processed} users")
    print(f"Users with enough data: {users_with_enough_data}")

    if not user_features:
        print("No users had sufficient data for feature calculation")
        return pd.DataFrame()

    # Convert to DataFrame
    features_df = pd.DataFrame(user_features)

    # Print churn statistics
    churn_rate = features_df['is_churned'].mean()
    print(f"Churn rate: {churn_rate:.3f} ({features_df['is_churned'].sum()}/{len(features_df)})")

    return features_df


def create_game_features_pipeline(game1_df, game2_plays_df, game3_df=None,
                                  observation_period_days=5, churn_prediction_days=10,
                                  consecutive_threshold_hours=24):
    """
    Create common features for all three games.

    Args:
        game1_df: Game 1 DataFrame
        game2_plays_df: Game 2 play events DataFrame
        game3_df: Game 3 DataFrame (optional)
        observation_period_days: Number of days for observation period
        churn_prediction_days: Number of days for churn prediction period
        consecutive_threshold_hours: Threshold for consecutive play detection
    """
    results = {}

    # Game 1 Features
    print("Calculating Game 1 common features...")
    try:
        game1_features = calculate_common_features(
            game1_df, 'game1', observation_period_days, churn_prediction_days, consecutive_threshold_hours
        )
        results['game1_features'] = game1_features
    except Exception as e:
        print(f"Error processing Game 1: {e}")
        results['game1_features'] = pd.DataFrame()

    # Game 2 Features (only for play events)
    print("Calculating Game 2 common features...")
    try:
        if not game2_plays_df.empty:
            game2_features = calculate_common_features(
                game2_plays_df, 'game2', observation_period_days, churn_prediction_days, consecutive_threshold_hours
            )
        else:
            game2_features = pd.DataFrame()
        results['game2_features'] = game2_features
    except Exception as e:
        print(f"Error processing Game 2: {e}")
        results['game2_features'] = pd.DataFrame()

    # Game 3 Features (if provided)
    if game3_df is not None:
        print("Calculating Game 3 common features...")
        try:
            game3_features = calculate_common_features(
                game3_df, 'game3', observation_period_days, churn_prediction_days, consecutive_threshold_hours
            )
            results['game3_features'] = game3_features
        except Exception as e:
            print(f"Error processing Game 3: {e}")
            results['game3_features'] = pd.DataFrame()

    return results


def main():
    """
    Main function with better error handling and debugging.
    """
    try:
        # Game 1
        print("Loading Game 1 data...")
        game1_df = pd.read_csv('rawdata_game1.csv', dtype={'device': str})
        print(f"Game 1 loaded: {game1_df.shape}")

        # Game 2 - assuming you have processed play events
        print("Loading Game 2 data...")
        game2_plays_df = pd.read_csv('play_events.csv')
        print(f"Game 2 loaded: {game2_plays_df.shape}")

        # FIXED: Use numeric values for observation and churn prediction periods
        observation_period_days = 5
        churn_prediction_days = 10

        print(f"Observation period: {observation_period_days} days")
        print(f"Churn prediction period: {churn_prediction_days} days")

        # Calculate common features for all games
        all_features = create_game_features_pipeline(
            game1_df=game1_df,
            game2_plays_df=game2_plays_df,
            game3_df=None,  # Add game3_df if you have it
            observation_period_days=observation_period_days,
            churn_prediction_days=churn_prediction_days,
            consecutive_threshold_hours=24
        )

        # Display results
        for game_name, features_df in all_features.items():
            print(f"\n{game_name.upper()} FEATURES:")
            print(f"Shape: {features_df.shape}")
            if not features_df.empty:
                print(features_df.head())

                # Save to CSV
                output_file = f"{game_name}_v2.csv"
                features_df.to_csv(output_file, index=False)
                print(f"Saved to {output_file}")
            else:
                print("No data available")

        # Merge purchase data with game2 features (if both files exist)
        try:
            print("\nAttempting to merge purchase data with game2 features...")
            df_main = pd.read_csv('game2_features_v2.csv')  # Updated filename
            df_purchase = pd.read_csv('user_level_aggregates.csv')

            # Ensure 'device.id' is treated as string
            df_main['device.id'] = df_main['device.id'].astype(str)
            df_purchase['device.id'] = df_purchase['device.id'].astype(str)

            # Subset to only necessary columns
            df_purchase_subset = df_purchase[['device.id', 'highestPrice', 'purchaseCount']]

            # Merge on 'device.id'
            df_merged = df_main.merge(df_purchase_subset, on='device.id', how='left')

            # Optional: fill NaNs if needed
            df_merged['highestPrice'] = df_merged['highestPrice'].fillna(0)
            df_merged['purchaseCount'] = df_merged['purchaseCount'].fillna(0)

            # Export the merged DataFrame to CSV
            df_merged.to_csv('game2_merged_output.csv', index=False)
            print("Successfully merged and saved game2_merged_output.csv")

        except FileNotFoundError as e:
            print(f"Could not merge purchase data: {e}")
            print("This is not critical - continuing without merge.")

        # Print summary statistics
        print("\n" + "=" * 50)
        print("SUMMARY STATISTICS")
        print("=" * 50)

        for game_name, features_df in all_features.items():
            if not features_df.empty:
                print(f"\n{game_name.upper()}:")
                numeric_cols = features_df.select_dtypes(include=[np.number]).columns
                print(features_df[numeric_cols].describe())

    except FileNotFoundError as e:
        print(f"File not found: {e}")
        print("Please ensure all required CSV files are in the current directory.")
    except Exception as e:
        print(f"Error during processing: {e}")
        print("Full error details:")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()