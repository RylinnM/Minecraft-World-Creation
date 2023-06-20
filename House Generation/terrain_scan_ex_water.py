# Written by: @R.Ma Copyright 2023
# MGAIA course 2023, LIACS, Leiden University

import logging
from random import randint
import numpy as np
from termcolor import colored

from gdpc import Block, Editor
from gdpc import geometry as geo
from gdpc import minecraft_tools as mt
from gdpc import editor_tools as et

logging.basicConfig(format=colored("%(name)s - %(levelname)s - %(message)s", color="yellow"))

ED = Editor(buffering=True)

building_area = ED.getBuildArea()  # BUILDAREA
world_slice = ED.loadWorldSlice(building_area.toRect(), cache=True)

starting_x, starting_y, starting_z = building_area.begin
ending_x, ending_y, ending_z = building_area.last

build_height = abs(starting_y - ending_y)
build_width = abs(starting_z - ending_z)
build_length = abs(starting_x - ending_x)

house_height = 10
house_width = 8
house_length = 20

buildRect = building_area.toRect()
heightmap = world_slice.heightmaps["MOTION_BLOCKING_NO_LEAVES"]
averageHeight = int(np.mean(heightmap))


def scanTerrain(area=building_area, househeight=house_height, housewidth=house_width, houselength=house_length):
    print("Scanning terrain...")

    min_diff = np.inf
    max_diff = 0
    flattest_subgrid = None
    roughest_subgrid = None
    counter = 0
    valid_counter = 0
    x1, y1, z1 = building_area.begin
    x2, y2, z2 = building_area.last
    print(f"Starting coordinates of scans: {x1, y1, z1}")
    print(f"Ending coordinates of scans:{x2, y2, z2}")

    flattest_subgrid_x, flattest_subgrid_y, flattest_subgrid_z = 0, 0, 0
    roughest_subgrid_x, roughest_subgrid_y, roughest_subgrid_z = 0, 0, 0
    for x in range(0, build_length - houselength):
        for z in range(0, build_width - housewidth):
            subgrid = np.zeros((houselength, housewidth))
            subgrid_valid = True
            for i in range(houselength):
                for j in range(housewidth):
                    #print(subgrid[i,j])
                    #print(heightmap[x+i,z+j])
                    subgrid[i,j] = heightmap[x+i,z+j]
                    y_ij = int(heightmap[x+i,z+j])
                    #print(f"Block at {starting_x+x+i, y_ij-1, starting_z+z+j} is {ED.getBlock((starting_x+ x + i, y_ij-1, starting_z+ z + j))}")
                    if "minecraft:water" in ED.getBlock((starting_x+ x + i, y_ij-1, starting_z+ z + j)).__str__():
                        subgrid_valid = False
                        break
                if subgrid_valid == False:
                    break
            counter += 1
            if subgrid_valid == False:
                continue
            else:
                valid_counter += 1

                avg_height = np.mean(subgrid)
                diff = np.abs(subgrid - avg_height)

                avg_diff = np.mean(diff)

                if avg_diff < min_diff:
                    min_diff = avg_diff
                    flattest_subgrid = subgrid
                    flattest_subgrid_x = x
                    flattest_subgrid_y = int(np.mean(subgrid))
                    flattest_subgrid_z = z

                if avg_diff > max_diff:
                    max_diff = avg_diff
                    roughest_subgrid = subgrid

                    roughest_subgrid_x = x
                    roughest_subgrid_y = heightmap[x,z]
                    roughest_subgrid_z = z
            # return the flattest subgrid and the starting coordinates of the subgrid

    print(f"{counter} grids scanned, {valid_counter} grids are valid and {counter-valid_counter} grids contain water surfaces")
    if flattest_subgrid is not None:
        print("The height map regarding the flattest subgrid in build area is:")
        print(flattest_subgrid)
        print("Average height:", np.mean(flattest_subgrid))
        print("Average difference:", min_diff)
        decision = True
    else:
        print("No suitable subgrid found within the given build area.")
        decision = False


    """
    print("The height map regarding the roughest subgrid in build area:")
    print(roughest_subgrid)
    print("Average height:", np.mean(roughest_subgrid))
    print("Average difference:", max_diff)
    print("Starting coordinates of roughest subgrid:")
    print(x1 + roughest_subgrid_x, y1+ roughest_subgrid_y, z1+ roughest_subgrid_z)
    """
    return flattest_subgrid_x + x1, flattest_subgrid_y, flattest_subgrid_z + z1, decision

def main():
    try:
        scanTerrain()
        """
        test = ED.getBlock((-97, 62, -48))
        print(test)
        if "minecraft:water" in test.__str__():
            print("Water detected")
        else:
            print("No water detected")
        """
        print("Done!")

    except KeyboardInterrupt:  # useful for aborting a run-away program
        print("Pressed Ctrl-C to kill program.")

if __name__ == '__main__':
    main()