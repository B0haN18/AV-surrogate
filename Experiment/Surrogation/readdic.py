import ast

def read_log_file(file_path):
    log_data = []
    with open(file_path, 'r') as file:
        for line in file:
            try:
                log_entry = ast.literal_eval(line.strip())  # Evaluate each line as a dictionary
                log_data.append(log_entry)  # Append the log entry to the list

            except (ValueError, SyntaxError):
                print(f"Skipping invalid log entry: {line}")

    return log_data

# Example usage
log_file_path = 'res.log'
log_dictionary = read_log_file(log_file_path)

# Now you have a list of dictionaries


sorted_list = sorted(log_dictionary, key=lambda x: x['fitness'])

for entry in sorted_list:
    print(entry)
