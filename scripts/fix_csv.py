import pandas as pd

# Load the CSV file
bbox_df = pd.read_csv("/users/gm00051/ChestX-ray14/labels/BBox_List_2017.csv")

# Rename columns to standard names
bbox_df = bbox_df.rename(columns={
    'Bbox [x': 'Bbox_x',
    'y': 'Bbox_y',
    'w': 'Bbox_w',
    'h]': 'Bbox_h'
})

# Drop unnecessary columns
bbox_df = bbox_df.drop(columns=['Unnamed: 6', 'Unnamed: 7', 'Unnamed: 8'])

# Validate bounding boxes
invalid_boxes = bbox_df[(bbox_df['Bbox_w'] <= 0) | (bbox_df['Bbox_h'] <= 0)]
if not invalid_boxes.empty:
    print("Warning: Found invalid bounding boxes:")
    print(invalid_boxes)

# Save the cleaned CSV for further use
bbox_df.to_csv("/users/gm00051/ChestX-ray14/labels/BBox_List_2017.csv", index=False)

