# handle game 2 specific features (purchasing related)

import json
import pandas as pd
from datetime import datetime


def preprocess_game_data(file_path):
    """
    Preprocess game event data for Table 3 analysis.

    Args:
        file_path (str): Path to the JSON Lines file

    Returns:
        dict: Contains processed DataFrames for different event types
    """

    # Lists to store processed records
    all_events = []
    purchase_events = []
    play_events = []

    # Read and parse JSON Lines file
    with open(file_path, 'r') as file:
        for line_num, line in enumerate(file, 1):
            try:
                # Parse JSON line
                event = json.loads(line.strip())

                # Extract basic attributes for Table 3
                processed_event = {
                    'device.id': event.get('uid'),  # Unique device id assigned by smartphone manufacturer
                    'time': event.get('time'),  # End time of the event
                    'event': event.get('event'),  # Type of this record (play or purchase)
                    'score': None,  # Score of the play (when event is play)
                    'purchase.price': None  # Purchase price (when event is purchase)
                }

                # Handle event-specific properties
                properties = event.get('properties', {})

                if event.get('event') == 'softPurchase':
                    # Purchase event processing - fill purchase.price
                    processed_event['purchase.price'] = properties.get('price')
                    processed_event['score'] = 0
                    purchase_events.append(processed_event.copy())

                else:  # event.get('event') == 'progress'
                    # Play/progress event processing - fill score
                    processed_event['event'] = 'play'
                    if properties.get('reward') is None:
                        processed_event['score'] = 0
                    else:
                        processed_event['score'] = properties.get('reward')
                    processed_event['purchase.price'] = 0

                    # processed_event['score'] = properties.get('reward')  # Using reward as score
                    play_events.append(processed_event.copy())

                all_events.append(processed_event)

            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                continue

    # Convert to DataFrames
    df_all = pd.DataFrame(all_events)
    df_purchases = pd.DataFrame(purchase_events)
    df_plays = pd.DataFrame(play_events)

    return {
        'all_events': df_all,
        'purchases': df_purchases,
        'plays': df_plays
    }


def calculate_game2_features(df_purchases):
    """
    Calculate Game #2 specific features as mentioned in your requirements.

    Args:
        df_purchases (DataFrame): Purchase events DataFrame

    Returns:
        DataFrame: Processed features for Game #2
    """

    if df_purchases.empty:
        return pd.DataFrame()

    # Sort by device.id and time to get purchase sequence
    df_sorted = df_purchases.sort_values(['device.id', 'time'])

    # Group by device to calculate features per user
    game2_features = []

    for device_id, group in df_sorted.groupby('device.id'):
        # Reset index for this user's purchases
        group = group.reset_index(drop=True)

        # Calculate relative time (t_pur_j) - time from first purchase
        first_purchase_time = group.iloc[0]['time']
        group['t_pur_j'] = group['time'] - first_purchase_time

        # Add purchase sequence number (j)
        group['purchase_seq'] = range(1, len(group) + 1)

        # Rename price column to p_j for clarity
        group['p_j'] = group['purchase.price']

        game2_features.append(group)

    # Combine all users
    if game2_features:
        result_df = pd.concat(game2_features, ignore_index=True)
        return result_df
    else:
        game2_df = pd.DataFrame()
        game2_df['time'] = pd.to_datetime(game2_df['time'], unit='s')  # time is in seconds
        return game2_df


def calculate_user_level_features(df_purchases, observation_period_end=None):
    """
    Calculate user-level aggregate features for Game #2.

    Args:
        df_purchases (DataFrame): Purchase events DataFrame
        observation_period_end (int): End time of observation period (top). If None, uses max time.

    Returns:
        DataFrame: User-level features with one row per user
    """

    if df_purchases.empty:
        return pd.DataFrame()

    # Determine observation period end time
    if observation_period_end is None:
        observation_period_end = df_purchases['time'].max()

    # Filter purchases within observation period
    df_op = df_purchases[df_purchases['time'] < observation_period_end].copy()

    if df_op.empty:
        return pd.DataFrame()

    # Calculate user-level aggregates
    user_features = df_op.groupby('device.id').agg({
        'purchase.price': ['max', 'count'],  # highestPrice and purchaseCount
        'time': ['min', 'max']  # first and last purchase times
    }).reset_index()

    # Flatten column names
    user_features.columns = ['device.id', 'highestPrice', 'purchaseCount', 'first_purchase_time', 'last_purchase_time']

    # Add observation period info
    user_features['observation_period_end'] = observation_period_end

    return user_features


# Example usage
def main():
    """
    Main function to demonstrate usage
    """

    # Process the data
    file_path = "part-00000"  # Replace with your file path

    try:
        # Preprocess all data
        processed_data = preprocess_game_data(file_path)

        print("Data preprocessing completed!")
        print(f"Total events: {len(processed_data['all_events'])}")
        print(f"Purchase events: {len(processed_data['purchases'])}")
        print(f"Play events: {len(processed_data['plays'])}")

        # Calculate Game #2 features
        game2_features = calculate_game2_features(processed_data['purchases'])
        print(f"\nGame #2 features calculated for {len(game2_features)} purchases")

        # Calculate user-level aggregates for Game #2
        user_aggregates = calculate_user_level_features(processed_data['purchases'])
        print(f"User-level aggregates calculated for {len(user_aggregates)} users")

        # Display sample results
        if not processed_data['purchases'].empty:
            print("\nSample Purchase Data:")
            print(processed_data['purchases'][['device.id', 'time', 'event', 'purchase.price']].head())

        if not processed_data['plays'].empty:
            print("\nSample Play Data:")
            print(processed_data['plays'][['device.id', 'time', 'event', 'score']].head())

        if not user_aggregates.empty:
            print("\nSample User-Level Aggregates:")
            print(user_aggregates[['device.id', 'highestPrice', 'purchaseCount']].head())

        # Save processed data
        processed_data['all_events'].to_csv('all_events.csv', index=False)
        processed_data['purchases'].to_csv('purchase_events.csv', index=False)
        processed_data['plays'].to_csv('play_events.csv', index=False)

        if not game2_features.empty:
            game2_features.to_csv('game2_features.csv', index=False)
        if not user_aggregates.empty:
            user_aggregates.to_csv('user_level_aggregates.csv', index=False)

        print("\nProcessed data saved to CSV files!")

    except FileNotFoundError:
        print(f"File {file_path} not found. Please check the file path.")
    except Exception as e:
        print(f"Error during processing: {e}")


if __name__ == "__main__":
    main()