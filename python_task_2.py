Question 1:
import pandas as pd

def calculate_distance_matrix(dataset_path):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(dataset_path)

    # Create a DataFrame with 'id' as index and columns
    distance_matrix = pd.DataFrame(index=df['id'].unique(), columns=df['id'].unique())

    # Set diagonal values to 0
    distance_matrix = distance_matrix.fillna(0)

    # Iterate over the rows of the DataFrame to calculate cumulative distances
    for _, row in df.iterrows():
        source_id = row['source']
        dest_id = row['destination']
        distance = row['distance']

        # Update the cumulative distances for bidirectional routes
        distance_matrix.at[source_id, dest_id] += distance
        distance_matrix.at[dest_id, source_id] += distance

    return distance_matrix

# Replace 'dataset-3.csv' with the actual path to the dataset file
result_dataframe = calculate_distance_matrix('dataset-3.csv')

# Display the result DataFrame
print(result_dataframe)

Question 2:

import pandas as pd

def unroll_distance_matrix(input_dataframe):
    # Get the column and index names from the input DataFrame
    ids = input_dataframe.index

    # Create an empty DataFrame to store unrolled distances
    unrolled_distances = pd.DataFrame(columns=['id_start', 'id_end', 'distance'])

    # Iterate through the combinations of id_start and id_end
    for id_start in ids:
        for id_end in ids:
            # Skip combinations where id_start is equal to id_end
            if id_start != id_end:
                # Get the distance value from the input DataFrame
                distance = input_dataframe.loc[id_start, id_end]

                # Append the values to the unrolled_distances DataFrame
                unrolled_distances = unrolled_distances.append({
                    'id_start': id_start,
                    'id_end': id_end,
                    'distance': distance
                }, ignore_index=True)

    return unrolled_distances

# Assuming result_dataframe is the DataFrame obtained from the calculate_distance_matrix function in Question 2
# Replace it with the actual DataFrame variable
result_dataframe = calculate_distance_matrix('dataset-3.csv')

# Apply the function and display the result
unrolled_dataframe = unroll_distance_matrix(result_dataframe)
print(unrolled_dataframe)

Question 3:
import pandas as pd

def find_ids_within_ten_percentage_threshold(input_dataframe, reference_value):
    # Filter rows based on the reference value
    reference_rows = input_dataframe[input_dataframe['id_start'] == reference_value]

    # Calculate the average distance for the reference value
    avg_distance = reference_rows['distance'].mean()

    # Define the threshold as 10% of the average distance
    threshold = 0.1 * avg_distance

    # Filter rows where the distance is within the 10% threshold
    selected_rows = input_dataframe[
        (input_dataframe['id_start'] != reference_value) &  # Exclude the reference value itself
        (input_dataframe['distance'] >= avg_distance - threshold) &
        (input_dataframe['distance'] <= avg_distance + threshold)
    ]

    # Get the unique values from the 'id_start' column and sort them
    result_list = sorted(selected_rows['id_start'].unique())

    return result_list

# Assuming unrolled_dataframe is the DataFrame obtained from the unroll_distance_matrix function in the previous step
# Replace it with the actual DataFrame variable 
unrolled_dataframe = unroll_distance_matrix(result_dataframe)

# Example: Find IDs within 10% threshold for the reference value 123
reference_value = 123
result_list = find_ids_within_ten_percentage_threshold(unrolled_dataframe, reference_value)

# Display the result list
print(result_list)

Question 4:
import pandas as pd

def calculate_toll_rate(input_dataframe):
    df = input_dataframe.copy()

    # Define rate coefficients for each vehicle type
    rate_coefficients = {'moto': 0.8, 'car': 1.2, 'rv': 1.5, 'bus': 2.2, 'truck': 3.6}

    # Calculate toll rates for each vehicle type
    for vehicle_type, rate_coefficient in rate_coefficients.items():
        df[vehicle_type] = df['distance'] * rate_coefficient

    return df

# Assuming unrolled_dataframe is the DataFrame obtained from the unroll_distance_matrix function in Question 3
# Replace it with the actual DataFrame variable
unrolled_dataframe = unroll_distance_matrix(result_dataframe)

# Apply the function and display the result
toll_rate_dataframe = calculate_toll_rate(unrolled_dataframe)
print(toll_rate_dataframe)

Question 5:
import pandas as pd
from datetime import datetime, timedelta, time

def calculate_time_based_toll_rates(input_dataframe):
    df = input_dataframe.copy()

    # Define time ranges and corresponding discount factors
    time_ranges = [
        ((0, 0, 0), (10, 0, 0), 0.8),
        ((10, 0, 0), (18, 0, 0), 1.2),
        ((18, 0, 0), (23, 59, 59), 0.8)
    ]

    # Define a constant discount factor for weekends
    weekend_discount_factor = 0.7

    # Create new columns for start_day, start_time, end_day, and end_time
    df['start_day'] = df['startDay'].apply(lambda x: datetime.strptime(str(x), '%Y-%m-%d').strftime('%A'))
    df['end_day'] = df['endDay'].apply(lambda x: datetime.strptime(str(x), '%Y-%m-%d').strftime('%A'))
    df['start_time'] = df.apply(lambda row: time(row['startTime'] // 100, row['startTime'] % 100, 0), axis=1)
    df['end_time'] = df.apply(lambda row: time(row['endTime'] // 100, row['endTime'] % 100, 0), axis=1)

    # Calculate toll rates based on time intervals
    for time_range in time_ranges:
        start_time, end_time, discount_factor = time_range
        mask = ((df['start_day'].isin(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']) &
                 (df['start_time'] >= time(*start_time)) & (df['start_time'] < time(*end_time))) |
                (df['start_day'].isin(['Saturday', 'Sunday'])))
        df.loc[mask, ['moto', 'car', 'rv', 'bus', 'truck']] *= discount_factor

    # Apply a constant discount factor for weekends
    weekend_mask = df['start_day'].isin(['Saturday', 'Sunday'])
    df.loc[weekend_mask, ['moto', 'car', 'rv', 'bus', 'truck']] *= weekend_discount_factor

    return df

# Assuming toll_rate_dataframe is the DataFrame obtained from the calculate_toll_rate function in Question 4
# Replace it with the actual DataFrame variable
toll_rate_dataframe = calculate_toll_rate(unrolled_dataframe)

# Apply the function and display the result
time_based_toll_rates_dataframe = calculate_time_based_toll_rates(toll_rate_dataframe)
print(time_based_toll_rates_dataframe)


