import os

def convert_tle_to_lis(input_file: str):
    """
    Converts TLE data from a text file to a .lis formatted file.

    Args:
        input_file (str): Name of the input file containing TLE data.
    """
    try:
        #Output file name: use the name of the input file, but with appendix .lis instead of .tle
        output_file = input_file.rsplit('.', 1)[0] + '.lis'

        with open(input_file, 'r') as infile:
            lines = infile.readlines()

        objects = []
        counter = 1

        for i in range(0, len(lines), 3):  # Process every 3 lines
            if len(lines[i:i+3]) < 3:  # Ignore incomplete TLE sets
                break
            name = f"Object_{counter:05d}" 
            satno = lines[i+1][2:7].strip()  # Satellite number
            cospar_id = f"19{satno[:2]}-{satno[2:]}A"  # COSPAR-like ID

            objects.append(f"{cospar_id:<12} {name:<25} {satno:<8} 1.00    0.10    0") #diameter = 1, albedo = 0.1, 0 = sphere (always!!)
            counter += 1

        # Write to output file, take header from example
        with open(output_file, 'w') as outfile:
            outfile.write("#-----------------------------P-R-O-O-F----------------------------------\n")
            outfile.write("#-----------PROGRAMM FOR RADAR AND OPTICAL OBSERVATION FORECASTING-------\n")
            outfile.write("# THIS IS A SUBSET OF THE CROSS REFERENCE FILE FOR DETERMINISTIC (TLE) OBJECTS\n")
            outfile.write("#COSPAR ID      NAME        SATNO      DIAMETER    ALBEDO     SHAPE\n")
            outfile.write("# [-]           [-]          [-]         [-]         [-]   sphere/plate (0/1)\n")
            outfile.write("\n".join(objects))

        print(f"File '{output_file}' has been created successfully.")

    except Exception as e:
        print(f"An error occurred: {e}") 
        

def convert_all_tle_from_folder(folder: str):
    """
    Converts all TLE files in a specified folder to .lis formatted files.

    Args:
        folder (str): Path to the folder containing .tle files.
    """
    try:
        files = os.listdir(folder)
        
        tle_files = [f for f in files if f.lower().endswith('.tle')]
        
        if not tle_files:
            print("No .tle files found in the folder.")
            return

        for tle_file in tle_files:
            full_path = os.path.join(folder, tle_file)
            print(f"Processing file: {tle_file}")
            convert_tle_to_lis(full_path)

        print("All .tle files have been processed successfully.")
        
    except Exception as e:
        print(f"An error occurred: {e}")

convert_all_tle_from_folder("TLE_files")