#!/bin/bash

# Get the start date, end date, and interval (in minutes) as arguments
start_date="$1"
end_date="$2"
interval="$3"

# Check if the interval is provided and is a positive number
if [[ ! "$interval" =~ ^[0-9]+$ ]] || [[ "$interval" -le 0 ]]; then
    echo "Please provide a valid positive number for the interval in minutes."
    exit 1
fi

# Ensure the start date is earlier than the end date
if [[ "$start_date" > "$end_date" ]]; then
    echo "The start date must be earlier than the end date."
    exit 1
fi

# Convert the start and end dates from YYYYMMDDHHMM format to date objects in UTC
start_date_converted=$(date -u -d "${start_date:0:4}-${start_date:4:2}-${start_date:6:2} ${start_date:8:2}:${start_date:10:2}" +"%s")
end_date_converted=$(date -u -d "${end_date:0:4}-${end_date:4:2}-${end_date:6:2} ${end_date:8:2}:${end_date:10:2}" +"%s")

current_date="$start_date_converted"

# Loop through the date range and print each incremented time
while [[ "$current_date" -le "$end_date_converted" ]]; do
    # Convert the current date (in seconds) back to the desired format (YYYYMMDDHHMM)
    formatted_date=$(date -u -d "@$current_date" +"%Y%m%d%H%M")
    echo "$formatted_date"
    
    # Increment by the specified interval in minutes (in seconds)
    current_date=$((current_date + interval * 60))
done

