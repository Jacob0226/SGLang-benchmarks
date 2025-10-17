import json  
from collections import defaultdict  
import argparse  
  
# Read JSON Lines formatted data from a file  
def read_json_lines(file_path):  
    with open(file_path, 'r') as file:  
        return [json.loads(line.strip()) for line in file]  
  
# Find the median record for each group  
def find_median(records):  
    sorted_records = sorted(records, key=lambda x: x['median_e2e_latency_ms'])  
    mid_index = len(sorted_records) // 2  
    # If even number of records, choose the one that is just left of the middle  
    if len(sorted_records) % 2 == 0:  
        return sorted_records[mid_index - 1]  
    else:  
        return sorted_records[mid_index]  
  
# Main function  
def main():  
    # Set up command line argument parsing  
    parser = argparse.ArgumentParser(  
        description=(
            'This script processes a JSON Lines file to analyze performance data. '
            'It groups records by request rate and completion status, finds the median latency for each group, '
            'and outputs results for groups with at least 3 records.'
        )
    )
    parser.add_argument('file_path', type=str, help='Path to the JSON lines file that contains the performance data.')
  
    args = parser.parse_args()  
  
    # Read the file  
    data = read_json_lines(args.file_path)  
  
    # Group data by 'request_rate' and 'completed'  
    grouped_data = defaultdict(list)  
    for record in data:  
        key = (record['request_rate'], record['completed'])  
        grouped_data[key].append(record)  
  
    # Find the median record for each group, and filter groups with less than 3 records  
    median_records = [  
        find_median(records)   
        for key, records in grouped_data.items()   
        if len(records) >= 3  
    ]  
  
    # Sort records by 'completed' and 'request_rate'  
    sorted_records = sorted(median_records, key=lambda x: (x['completed'], x['request_rate']))  
  
    # Output results  
    for record in sorted_records:  
        print(f"Group (request_rate={record['request_rate']}, completed={record['completed']}):")  
        print(f" median_e2e_latency_ms: {record['median_e2e_latency_ms']}")  
        print(f" median_ttft_ms: {record['median_ttft_ms']}")  
        print(f" median_itl_ms: {record['median_itl_ms']}\n")  
  
if __name__ == '__main__':  
    main()  

