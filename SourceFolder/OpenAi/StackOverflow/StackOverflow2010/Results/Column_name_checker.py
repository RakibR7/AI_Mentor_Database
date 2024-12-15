import pandas as pd

# Load the CSV file
csv_file = 'CSV_V4.csv'
df = pd.read_csv(csv_file)

# Print the column names
print("Column Names:", df.columns.tolist())
