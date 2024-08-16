import os

def find_files_in_directory(file_names, search_directory):
    """Searches for files in the search_directory and its subdirectories.
    
    Args:
        file_names (list): List of file names to search for.
        search_directory (str): The directory to search within.
    
    Returns:
        dict: A dictionary where keys are file names and values are lists of paths where the files are found.
    """
    file_locations = {file_name: [] for file_name in file_names}
    
    for root, dirs, files in os.walk(search_directory):
        for file_name in files:
            if file_name in file_locations:
                file_locations[file_name].append(os.path.join(root, file_name))
    
    return file_locations

def read_file_names_from_directory(directory):
    """Reads all file names from a specified directory.
    
    Args:
        directory (str): The directory to read file names from.
    
    Returns:
        list: A list of file names in the directory.
    """
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

def main(source_directory, search_directory):
    """Main function to find and display file locations.
    
    Args:
        source_directory (str): The directory to read file names from.
        search_directory (str): The directory to search within.
    """
    # Check if source directory exists
    if not os.path.exists(source_directory) or not os.path.isdir(source_directory):
        print(f"Error: Source directory '{source_directory}' does not exist or is not a directory.")
        return
    else:
    	print('found 1')

    # Check if search directory exists
    if not os.path.exists(search_directory) or not os.path.isdir(search_directory):
        print(f"Error: Search directory '{search_directory}' does not exist or is not a directory.")
        return
    else:
    	print('found 2')

    
    file_names = read_file_names_from_directory(source_directory)
    file_locations = find_files_in_directory(file_names, search_directory)
    
    for file_name, locations in file_locations.items():
        if locations:
            print(f"File '{file_name}' found at:")
            for location in locations:
                print(f"  {location}")
        else:
            print(f"File '{file_name}' not found in the search directory.")

if __name__ == "__main__":
    # Update these paths to your directories
    source_directory = "/home/anonymous/code/tflite-micro/output/tflite-aut3/cc/"  # Directory to read file names from
    search_directory = "/home/anonymous/code/tflite-micro/tensorflow/lite/"  # Directory to search within
    
    
    main(source_directory, search_directory)

