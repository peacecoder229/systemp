import argparse
import pandas as pd
import numpy as np


def validate_data(original_df, new_df, index_colname, avg_lat_colname, group_by_col):
    # Re-index original_df for easy value access
    original_df.reset_index(inplace=True)
    
    # Iterate through each unique value in the group_by_col
    for all_tokens_value in original_df[group_by_col].unique():
        # Get the column name for this unique value in new_df
        column_name = f'{avg_lat_colname}_{all_tokens_value}'
        
        # For each value in the index column
        for idx_value in original_df[index_colname].unique():
            # Get the original avg_lat value
            original_value = original_df.loc[
                (original_df[index_colname] == idx_value) & (original_df[group_by_col] == all_tokens_value),
                avg_lat_colname
            ].values
            
            # Get the new avg_lat value from new_df
            new_value = new_df.loc[
                new_df.index == idx_value,
                column_name
            ].values

            #print(f"Original value for {index_colname} = {idx_value} and {group_by_col} = {all_tokens_value}: {original_value}")
            #print(f"New value in {column_name} for {index_colname} = {idx_value}: {new_value}")
            # Compare values (considering possible multiple matches and NaNs)
            if not np.array_equal(original_value, new_value):
                print(f"Mismatch found for {index_colname} = {idx_value} and {group_by_col} = {all_tokens_value}")



def process_file(input_file, output_file, index_colname, avg_lat_colname, group_by_col):
    # Load the data
    df = pd.read_csv(input_file, sep=',')
    df=df.set_index(index_colname)
    print(df.columns)
    df[group_by_col] = df[group_by_col].round(decimals=1)
    # Prepare a new DataFrame with the specified index column as the primary column
    unique_indices = df.index.unique()
    new_df = pd.DataFrame(index=unique_indices)
    # Sort the new DataFrame by the specified index column to ensure consistent ordering
    #new_df = new_df.sort_values(index_colname)

    # Iterate over each unique value in the specified group-by column
    for all_tokens_value in df[group_by_col].unique():
        # Filter the original DataFrame
        filtered_df = df[df[group_by_col] == all_tokens_value]
        #filtered_df = filtered_df.set_index(index_colname)
        # Create a new column for this unique group-by column value
        column_name = f'{avg_lat_colname}_{all_tokens_value}'
        #print(column_name)
        #print(filtered_df[avg_lat_colname])
        # Map avg_lat_colname values to the index column, ensuring alignment
        #mapped_values = filtered_df.set_index(index_colname)[avg_lat_colname].reindex(new_df[index_colname])
        
        # Add this as a new column to 'new_df'
        #new_df[column_name] = mapped_values
        #print(new_df.index)
        #print(filtered_df.index)
        new_df[column_name] = filtered_df[avg_lat_colname]

    # Save this DataFrame to the output CSV file
    new_df.sort_index(inplace=True)
    new_df.to_csv(output_file, index=True)
    return new_df, df
if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Process input CSV file to generate a new CSV file with transformed data.")
    parser.add_argument("--inputfile", required=True, help="Path to the input CSV file.")
    parser.add_argument("--outputfile", required=True, help="Path to the output CSV file.")
    parser.add_argument("--indexcol", required=True, help="Name of the index column.")
    parser.add_argument("--avglatcol", required=True, help="Column name for Avg_lat.")
    parser.add_argument("--groupbycol", required=True, help="Group-by column name, e.g., ALLTokens.")

    args = parser.parse_args()

    # Process the file based on the provided arguments
    new_df, df = process_file(args.inputfile, args.outputfile, args.indexcol, args.avglatcol, args.groupbycol)
    validate_data(df, new_df, args.indexcol, args.avglatcol, args.groupbycol)

