from pathlib import Path
import json

from mp_movement_classifier.utils.h36m_csv_converter import H36MConverter


converter = H36MConverter()
path = Path("../../data/MMpose/df_files_3d")

# Create output directory for BVH files
output_dir = Path("../../data/position_csv_files")
output_dir.mkdir(exist_ok=True)

with open('../../data/common_motion_mapping.json', 'r') as f:
    common_motion_mapping = json.load(f)['mapping']

# csv_file = "../../data/MMpose/filtered_csv_files/filtered_subject_12_motion_05.csv"
# bvh_file = "../../data/MMpose/filtered_csv_files/filtered_subject_12_motion_05.bvh"
# channels, header = converter.convert_to_bvh(csv_file, bvh_file)

for csv_file in path.glob("*.csv"):
    bvh_file = output_dir / csv_file.name.replace(".csv", ".bvh")
    out_file = output_dir / csv_file.name

    # print(f"Converting {csv_file.name} ")
    # try:
        # channels, header = converter.convert_to_bvh(str(csv_file), str(bvh_file))
        # print(len(channels[0]))
    # df = converter.convert_to_csv(str(csv_file), str(out_file))
    # converter.plot_pos_rep(str(csv_file), str(csv_file.name.replace(".csv", ".png")))
    df =converter.convert_position_to_csv(str(csv_file), str(out_file))

    # except Exception as e:
    #     print(f"Error converting {csv_file.name}: {str(e)}")

print("\nConversion complete!")