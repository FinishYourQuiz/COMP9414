counter=84
while [ $counter -le 84 ]
do
  counter_str=$counter
  while [ ${#counter_str} -ne 4 ];
    do
    counter_str="0"$counter_str
  done
  tmp="scene"$counter_str"_00"
  c:\\Users\\Delta\\GoogleDrive\\University_Work\\School_work\\COMP9491\\ScanNet\\venv39\\Scripts\\python.exe reader.py --filename ../../../scannet_data/scans/$tmp/$tmp.sens --output_path ../../../scannet_data/extracted/$tmp --export_depth_images --export_color_images --export_poses --export_intrinsics
((counter++))
done
# python reader.py --filename "G:/datasets_tmp/applied AI/scannet_data/full_scannet/scans/scene0039_00/scene0039_00.sens" --output_path "G:/datasets_tmp/applied AI/scannet_data/extracted/extracted/" --export_depth_images --export_color_images --export_poses --export_intrinsics