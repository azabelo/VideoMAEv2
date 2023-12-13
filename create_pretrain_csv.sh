#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <input_csv_file>"
  exit 1
fi

# Input CSV file
input_csv="$1"

# Check if the input file exists
if [ ! -f "$input_csv" ]; then
  echo "Error: Input file '$input_csv' not found."
  exit 1
fi

# Output CSV file
output_csv="official_pretrain.csv"

# Process each line in the input CSV file
count=0
while read -r video_path _; do
  # Extract frame count using ffprobe (assuming ffprobe is available)
  frame_count=$(ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of default=nokey=1:noprint_wrappers=1 "$video_path")

  # Check if frame_count is empty (indicating an error in ffprobe)
  if [ -z "$frame_count" ]; then
    echo "Error: 'N/A' encountered in $video_path. Skipping..."
    continue
  fi

  # Append the row to the output CSV file
  echo "$video_path $frame_count" >> "$output_csv"

  # Increment the counter
  ((count++))

  # Print the count every 100 videos
  if ((count % 100 == 0)); then
    echo "Processed $count videos."
  fi
done < "$input_csv"

echo "Processing complete. Output saved to '$output_csv'."
