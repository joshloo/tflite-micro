import os
import hashlib
import difflib

def compute_file_hash(file_path, hash_algo=hashlib.sha256):
    """Compute the hash of a file.
    
    Args:
        file_path (str): The path to the file.
        hash_algo (callable): The hash algorithm to use (default is SHA-256).
    
    Returns:
        str: The hexadecimal digest of the file's hash.
    """
    hash_obj = hash_algo()
    try:
        with open(file_path, 'rb') as file:
            while chunk := file.read(8192):
                hash_obj.update(chunk)
    except OSError as e:
        print(f"Error reading file {file_path}: {e}")
    return hash_obj.hexdigest()

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

def compute_file_content(file_path):
    """Read the content of a file.
    
    Args:
        file_path (str): The path to the file.
    
    Returns:
        str: The content of the file.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.readlines()

def dump_deltas(source_file_path, found_file_path, output_dir):
    """Dump the differences between two files.
    
    Args:
        source_file_path (str): Path to the source file.
        found_file_path (str): Path to the found file.
        output_dir (str): Directory where the delta files will be saved.
    """
    source_lines = compute_file_content(source_file_path)
    found_lines = compute_file_content(found_file_path)
    
    diff = difflib.unified_diff(
        source_lines, 
        found_lines, 
        fromfile=os.path.basename(source_file_path), 
        tofile=os.path.basename(found_file_path)
    )
    
    delta_file_path = os.path.join(output_dir, f"delta_{os.path.basename(source_file_path)}_{os.path.basename(found_file_path)}.diff")
    
    with open(delta_file_path, 'w', encoding='utf-8') as delta_file:
        delta_file.writelines(diff)
    
    print(f"Delta between '{source_file_path}' and '{found_file_path}' saved to '{delta_file_path}'.")

def main(source_directory, search_directory, output_directory):
    """Main function to find and display file locations with content comparison and delta dumping.
    
    Args:
        source_directory (str): The directory to read file names from.
        search_directory (str): The directory to search within.
        output_directory (str): Directory to save delta files.
    """
    if not os.path.exists(source_directory) or not os.path.isdir(source_directory):
        print(f"Error: Source directory '{source_directory}' does not exist or is not a directory.")
        return

    if not os.path.exists(search_directory) or not os.path.isdir(search_directory):
        print(f"Error: Search directory '{search_directory}' does not exist or is not a directory.")
        return

    if not os.path.exists(output_directory) or not os.path.isdir(output_directory):
        print(f"Error: Output directory '{output_directory}' does not exist or is not a directory.")
        return
    
    file_names = read_file_names_from_directory(source_directory)
    file_locations = find_files_in_directory(file_names, search_directory)
    
    for file_name, locations in file_locations.items():
        source_file_path = os.path.join(source_directory, file_name)
        if not os.path.isfile(source_file_path):
            print(f"File '{file_name}' does not exist in source directory.")
            continue
        
        source_file_hash = compute_file_hash(source_file_path)
        found_match = False

        for location in locations:
            found_file_hash = compute_file_hash(location)
            if source_file_hash == found_file_hash:
                print(f"File '{file_name}' found at {location} and matches the source file.")
                found_match = True
                break
        
        if not found_match:
            print(f"File '{file_name}' found but no exact match for content in the search directory.")
            for location in locations:
                dump_deltas(source_file_path, location, output_directory)

if __name__ == "__main__":
    # Update these paths to your directories
    source_directory = "/home/anonymous/code/tflite-micro/output/tflite-aut3/cc/"  # Directory to read file names from
    search_directory = "/home/anonymous/code/tflite-micro/tensorflow/lite/"  # Directory to search within
    output_directory = "/home/anonymous/code/tflite-micro/output/tflite-aut3/out/"  # Directory to save delta files
    
    main(source_directory, search_directory, output_directory)

