Task 1 
Question 1:

import pandas as pd

def generate_car_matrix(dataset_path):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(dataset_path)

    # Pivot the DataFrame using id_1 and id_2 columns
    result_df = df.pivot(index='id_1', columns='id_2', values='car')

    # Fill NaN values with 0 (for cells without corresponding car values)
    result_df = result_df.fillna(0)

    # Set diagonal values to 0
    result_df.values[[range(result_df.shape[0])]*2] = 0

    return result_df

# Replace 'dataset-1.csv' with the actual path to the dataset file
result_dataframe = generate_car_matrix('dataset-1.csv')

# Display the result DataFrame
print(result_dataframe)

Question 2:

import pandas as pd

def get_type_count(dataset_path):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(dataset_path)

    # Add a new categorical column 'car_type' based on 'car' values
    conditions = [
        (df['car'] <= 15),
        (df['car'] > 15) & (df['car'] <= 25),
        (df['car'] > 25)
    ]

    choices = ['low', 'medium', 'high']

    df['car_type'] = np.select(conditions, choices, default='unknown')

    # Calculate the count of occurrences for each 'car_type' category
    type_count = df['car_type'].value_counts().to_dict()

    # Sort the dictionary alphabetically based on keys
    type_count = dict(sorted(type_count.items()))

    return type_count

# Replace 'dataset-1.csv' with the actual path to the dataset file
result_dict = get_type_count('dataset-1.csv')

# Display the result dictionary
print(result_dict)

Question 3:
import pandas as pd

def get_bus_indexes(dataset_path):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(dataset_path)

    # Calculate the mean value of the 'bus' column
    bus_mean = df['bus'].mean()

    # Identify indices where 'bus' values are greater than twice the mean
    bus_indexes = df[df['bus'] > 2 * bus_mean].index.tolist()

    # Sort the indices in ascending order
    bus_indexes.sort()

    return bus_indexes

# Replace 'dataset-1.csv' with the actual path to the dataset file
result_list = get_bus_indexes('dataset-1.csv')

# Display the result list
print(result_list)

Question 4:

import pandas as pd

def filter_routes(dataset_path):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(dataset_path)

    # Calculate the average value of the 'truck' column for each route
    route_avg_truck = df.groupby('route')['truck'].mean()

    # Filter routes where the average of 'truck' values is greater than 7
    selected_routes = route_avg_truck[route_avg_truck > 7].index.tolist()

    # Sort the list of selected routes in ascending order
    selected_routes.sort()

    return selected_routes

# Replace 'dataset-1.csv' with the actual path to the dataset file
result_list = filter_routes('dataset-1.csv')

# Display the result list
print(result_list)

Question 5:
def multiply_matrix(input_dataframe):
    # Make a copy to avoid modifying the original DataFrame
    modified_df = input_dataframe.copy()

    # Apply the specified logic to each value in the DataFrame
    modified_df = modified_df.applymap(lambda x: x * 0.75 if x > 20 else x * 1.25)

    # Round values to 1 decimal place
    modified_df = modified_df.round(1)

    return modified_df

# Assuming result_dataframe is the DataFrame obtained from the generate_car_matrix function in Question 1
# Replace it with the actual DataFrame variable you have
result_dataframe = generate_car_matrix('dataset-1.csv')

# Apply the modification logic and display the result
modified_dataframe = multiply_matrix(result_dataframe)
print(modified_dataframe)

Question 6:
import pandas as pd

def check_timestamp_completeness(df):
    # Convert the 'timestamp' column to datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Extract day and time information
    df['day'] = df['timestamp'].dt.day_name()
    df['hour'] = df['timestamp'].dt.hour

    # Define expected days and hours
    expected_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    expected_hours = list(range(24))

    # Check completeness for each (id, id_2) pair
    completeness_series = df.groupby(['id', 'id_2']).apply(lambda group:
        all(day in group['day'].unique() for day in expected_days) and
        all(hour in group['hour'].unique() for hour in expected_hours)
    )

    return completeness_series

# Assuming df is your DataFrame from dataset-2.csv
# Replace it with the actual DataFrame variable  
df = pd.read_csv('dataset-2.csv')

# Apply the function and display the result
result_series = check_timestamp_completeness(df)
print(result_series)



