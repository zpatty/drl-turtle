# import os
# from stl import mesh
# import stl

# def stl_to_obj(input_file, output_file):
#     # Load the STL file
#     stl_mesh = mesh.Mesh.from_file(input_file)

#     # Save the mesh as OBJ
#     stl_mesh.save(output_file, mode=stl.Mode.ASCII)

# def convert_stl_files_to_obj(directory):
#     # Get a list of all STL files in the directory
#     stl_files = [file for file in os.listdir(directory) if file.endswith(".STL")]

#     # Create the output directory if it doesn't exist
#     output_directory = os.path.join(directory, "obj_files")
#     os.makedirs(output_directory, exist_ok=True)

#     # Convert each STL file to OBJ and save it
#     for stl_file in stl_files:
#         input_file = os.path.join(directory, stl_file)
#         output_file = os.path.join(output_directory, os.path.splitext(stl_file)[0] + ".obj")
#         stl_to_obj(input_file, output_file)
#         print(f"Converted {stl_file} to {os.path.basename(output_file)}")

# # Specify the directory containing the STL files
# stl_directory = "asset/meshes/visual"

# # Convert the STL files to OBJ
# convert_stl_files_to_obj(stl_directory)


def convert_stl_files_to_obj(directory):
    # Get a list of all STL files in the directory
    stl_files = [file for file in os.listdir(directory) if file.endswith(".stl")]

    # Create the output directory if it doesn't exist
    output_directory = os.path.join(directory, "obj_files")
    os.makedirs(output_directory, exist_ok=True)

    # Convert each STL file to OBJ and save it
    for stl_file in stl_files:
        input_file = os.path.join(directory, stl_file)
        output_file = os.path.join(output_directory, os.path.splitext(stl_file)[0] + ".obj")
        convertor_manual(input_file, output_file)
        print(f"Converted {stl_file} to {os.path.basename(output_file)}")



import os
from stl_obj_convertor.convertor import convertor_manual 

stl_directory = "asset/meshes/collision"

tst = stl_directory + '/base_link.STL' 
targ = stl_directory + '/base_link.stl' 


convert_stl_files_to_obj(stl_directory)
#print('d')


# import stl
# from stl import mesh
# binary_stl_files = [file for file in os.listdir(stl_directory) if file.endswith(".STL")]
# for b in binary_stl_files:
#     src = os.path.join(stl_directory, b)
#     targ = os.path.join(stl_directory, b.replace('STL', 'stl'))
#     your_mesh = mesh.Mesh.from_file(src)
#     your_mesh.save(targ, mode=stl.Mode.ASCII)