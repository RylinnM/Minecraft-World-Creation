# Written by: @R.Ma Copyright 2023
# MGAIA course 2023, LIACS, Leiden University
import logging
import numpy as np
from termcolor import colored

from gdpc import Block, Editor
from gdpc import geometry as geo
from gdpc import minecraft_tools as mt
from gdpc import editor_tools as et

from terrain_scan_ex_water import scanTerrain
from materials import material_selection
from structure import buildFoundation, buildRoof
from ext_deco import buildWindows, buildDoor, buildLights, clear_prem, base_rein,build_bell
from int_deco import applyCarpet, addFurniture, addintLights, addGarden, addSign, addbookshelf

logging.basicConfig(format=colored("%(name)s - %(levelname)s - %(message)s", color="yellow"))

ED = Editor(buffering=True)

building_area = ED.getBuildArea()  # BUILDAREA
buildRect = building_area.toRect()
world_slice = ED.loadWorldSlice(building_area.toRect(), cache=True)

starting_x, starting_y, starting_z = building_area.begin
ending_x, ending_y, ending_z = building_area.last

build_height = abs(starting_y - ending_y)
build_width = abs(starting_z - ending_z)
build_length = abs(starting_x - ending_x)

house_height = np.random.randint(4, 7)
house_width = np.random.randint(9, 15)
house_length = np.random.randint(14, 20)

heightmap = world_slice.heightmaps["MOTION_BLOCKING_NO_LEAVES"]
averageHeight = int(np.mean(heightmap))

house_x, house_y, house_z, decision = scanTerrain(area=building_area, househeight=house_height, housewidth=house_width, houselength=house_length)
def main():

    if decision == True:
        print("Building a house at: {}, {}, {}".format(house_x, house_y, house_z))
        print("The dimensions of the house are: {}x{}x{}".format(house_height, house_width, house_length))
        buildFoundation(house_x, house_y, house_z, house_height, house_width, house_length)
        base_rein(house_x, house_y, house_z, house_height, house_width, house_length)
        applyCarpet(house_x, house_y, house_z, house_height, house_width, house_length)
        buildRoof(house_x, house_y, house_z, house_height, house_width, house_length)
        clear_prem(house_x, house_y, house_z, house_height, house_width, house_length)
        buildWindows(house_x, house_y, house_z, house_height, house_width, house_length)
        buildDoor(house_x, house_y, house_z, house_height, house_width, house_length)
        build_bell(house_x, house_y, house_z, house_height, house_width, house_length)
        buildLights(house_x, house_y, house_z, house_height, house_width, house_length)
        addFurniture(house_x, house_y, house_z, house_height, house_width, house_length)
        addintLights(house_x, house_y, house_z, house_height, house_width, house_length)
        addGarden(house_x, house_y, house_z, house_height, house_width, house_length)
        addSign(house_x, house_y, house_z, house_height, house_width, house_length)
        print("house built with success!")
    else:
        print("Please relocate the building area as the terrain is not suitable for building a house.")

if __name__ == "__main__":
    main()
