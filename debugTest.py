import pickle

def check_pkl_file(file_path):
    try:
        # Try loading the file
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        
        # If no exception is raised, check the data's integrity (optional)
        print("File loaded successfully.")
        print("Data structure looks like:", type(data))  # This will give you the data type structure
        

        # Check the structure of the tuple
        print("Tuple contents:", data)

        # If it's a tuple, you can access its elements
        dataset, vocab, max_len = data

        # Print details about each part of the tuple
        print("Dataset:", dataset)
        print("Vocabulary:", vocab)
        print("Max Length:", max_len)


        return True
    except (pickle.UnpicklingError, EOFError) as e:
        # Handle known exceptions for corrupted files
        print(f"Error: {e}. The file is corrupt.")
        return False
    except Exception as e:
        # Catch other potential errors
        print(f"An unexpected error occurred: {e}")
        return False

# Check if the file is corrupt
file_path = "preprocessed_data.pkl"
if not check_pkl_file(file_path):
    print("The file could not be loaded, it may be corrupt.")
else:
    print("The file is valid.")
