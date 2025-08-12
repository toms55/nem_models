import os


# Path to the folder with your CSV files
folder_path = '/home/tom/Documents/nem_models/nem_data//'
output_file = os.path.join(folder_path, 'dispatch_history_2021-2025.csv')


# Get all CSV files in the folder and sort them alphabetically
csv_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.CSV')])

with open(output_file, 'w', encoding='utf-8') as outfile:
    for i, filename in enumerate(csv_files):
        filepath = os.path.join(folder_path, filename)
        with open(filepath, 'r', encoding='utf-8') as infile:
            lines = infile.readlines()
            # Write header only from the first file
            if i == 0:
                outfile.writelines(lines)
            else:
                outfile.writelines(lines[1:])  # skip header

print(f"Combined {len(csv_files)} files into '{output_file}'")
