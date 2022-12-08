import os
import pandas as pd
ANNOTATION_FRAME_PATH = "../data_turntable/annotated_frame"
obj = "0wheel2"
CSV_PATH = "../data_turntable/csv_data/"

def update_csv(csv_file, list_file_in_annotation_frames):
    df = pd.read_csv(csv_file)
    print(df.shape)
    df =  df[df.filename.isin(list_file_in_annotation_frames)]
    print(df.shape)
    df.to_csv(csv_file, index=False)


def main():
    list_annotated_frames = os.listdir(os.path.join(ANNOTATION_FRAME_PATH, obj))
    csv_file = obj + ".csv"
    csv_file_path = os.path.join(CSV_PATH,csv_file )
    update_csv(csv_file_path, list_annotated_frames)

if __name__ == '__main__':
    main()
