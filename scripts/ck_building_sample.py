import json
import os
import pathlib # More modern way to handle paths

def count_building_damage(json_directory):
    """
    Counts building damage subtypes from JSON files in a specified directory.

    Args:
        json_directory (str): The path to the directory containing the JSON files.

    Returns:
        tuple: A tuple containing:
               - dict: A dictionary with counts for each subtype.
               - int: The total number of buildings counted.
               - list: A list of files that caused errors during processing.
    """
    damage_counts = {
        "no-damage": 0,
        "minor-damage": 0,
        "major-damage": 0,
        "destroyed": 0,
        "unknown_subtype": 0 # Optional: To count features with unexpected subtypes
    }
    total_buildings = 0
    error_files = []

    # Convert the input string path to a Path object
    dir_path = pathlib.Path(json_directory)

    if not dir_path.is_dir():
        print(f"Error: Directory not found: {json_directory}")
        return None, 0, []

    print(f"Scanning directory: {dir_path}")

    # Iterate through files in the specified directory
    for file_path in dir_path.glob('*.json'): # Use glob to easily find .json files
        print(f"Processing file: {file_path.name}...")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

                # Navigate to the features list (using 'lng_lat' as per example)
                # Add checks to ensure keys exist
                if 'features' in data and isinstance(data['features'], dict) and \
                   'lng_lat' in data['features'] and isinstance(data['features']['lng_lat'], list):

                    buildings = data['features']['lng_lat']

                    for building in buildings:
                        # Check structure for properties and subtype
                        if 'properties' in building and isinstance(building['properties'], dict) and \
                           'subtype' in building['properties']:

                            subtype = building['properties'].get('subtype', '').lower() # Get subtype, default to empty string, lowercase

                            # Increment the appropriate counter
                            if subtype in damage_counts:
                                damage_counts[subtype] += 1
                            elif subtype: # If subtype exists but is not one of the expected ones
                                print(f"  - Found unexpected subtype '{subtype}' in {file_path.name}")
                                damage_counts["unknown_subtype"] += 1
                            # else: # Subtype key exists but value is empty or None - implicitly ignored

                            # Only increment total if we found a relevant subtype
                            if subtype in damage_counts or subtype:
                                 total_buildings += 1

                        # Optional: Add warnings for malformed building entries if needed
                        # else:
                        #    print(f"  - Warning: Skipping feature in {file_path.name} due to missing 'properties' or 'subtype'.")

                else:
                     print(f"  - Warning: Skipping file {file_path.name} due to missing 'features' or 'features['lng_lat']' list.")


        except json.JSONDecodeError:
            print(f"  - Error: Could not decode JSON from {file_path.name}.")
            error_files.append(file_path.name)
        except Exception as e:
            print(f"  - Error: An unexpected error occurred processing {file_path.name}: {e}")
            error_files.append(file_path.name)

    return damage_counts, total_buildings, error_files

# --- Main Execution ---
if __name__ == "__main__":
    # Get the directory path from the user
    # Replace "path/to/your/json/folder" with the actual path or leave as is for user input
    json_folder_path = input("Enter the path to the folder containing your JSON files: ")
    # Example: json_folder_path = "/Users/you/data/damage_jsons"
    # Or for Windows: json_folder_path = r"C:\Users\you\data\damage_jsons" # Use raw string

    counts, total, errors = count_building_damage(json_folder_path)

    if counts is not None: # Check if counting was successful
        print("\n--- Damage Assessment Summary ---")
        print(f"No Damage:     {counts.get('no-damage', 0)}")
        print(f"Minor Damage:  {counts.get('minor-damage', 0)}")
        print(f"Major Damage:  {counts.get('major-damage', 0)}")
        print(f"Destroyed:     {counts.get('destroyed', 0)}")
        if counts.get('unknown_subtype', 0) > 0:
             print(f"Unknown Subtype: {counts.get('unknown_subtype', 0)}")
        print("---------------------------------")
        print(f"Total Buildings Assessed: {total}")
        print("---------------------------------")

        if errors:
            print("\nWarning: The following files encountered errors during processing:")
            for filename in errors:
                print(f"- {filename}")