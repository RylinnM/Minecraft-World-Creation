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

from materials import material_selection

logging.basicConfig(format=colored("%(name)s - %(levelname)s - %(message)s", color="yellow"))

ED = Editor(buffering=True)

building_area = ED.getBuildArea()  # BUILDAREA

world_slice = ED.loadWorldSlice(building_area.toRect(), cache=True)

starting_x, starting_y, starting_z = building_area.begin
ending_x, ending_y, ending_z = building_area.last

build_height = abs(starting_y - ending_y)
build_width = abs(starting_z - ending_z)
build_length = abs(starting_x - ending_x)

buildRect = building_area.toRect()
heightmap = world_slice.heightmaps["MOTION_BLOCKING_NO_LEAVES"]
averageHeight = int(np.mean(heightmap))


def buildFoundation(house_x, house_y, house_z, house_height, house_width, house_length):
    print("building foundation...")
    material = material_selection("foundation")
    for i in range(0, house_length):
        for j in range(0, house_width):
            for k in range(1):
                ED.placeBlock((house_x + i, house_y + k, house_z + j), Block(material))
    print("foundation built with {}".format(material))

    # build the walls of the house with the interior space
    print("building walls...")
    material = material_selection("wall")
    for x in range(0, house_length):
        for z in range(0, house_width):
            for y in range(1, house_height):
                if x == 0 or x == house_length - 1 or z == 0 or z == house_width - 1:
                    ED.placeBlock((house_x + x, house_y + y, house_z + z), Block(material))
                else:
                    ED.placeBlock((house_x + x, house_y + y, house_z + z), Block("minecraft:air"))
    print("walls built with {}".format(material))


def buildRoof(house_x, house_y, house_z, house_height, house_width, house_length):
    print("building roof...")
    material = material_selection("roof")
    for x in range(0, house_length):
        for z in range(0, house_width):
            for y in range(1):
                if x == 0 or x == house_length - 1 or z == 0 or z == house_width - 1:
                    ED.placeBlock((house_x + x, house_y + house_height + y, house_z + z), Block(material))
                else:
                    ED.placeBlock((house_x + x, house_y + house_height + y, house_z + z), Block("minecraft:air"))

    for x in range(0, house_length):
        for z in range(0, house_width):
            for y in range(1):
                if x == 1 or x == house_length - 2 or z == 1 or z == house_width - 2:
                    ED.placeBlock((house_x + x, house_y + house_height + y + 1, house_z + z), Block(material))
                else:
                    ED.placeBlock((house_x + x, house_y + house_height + y + 1, house_z + z), Block("minecraft:air"))

    for x in range(0, house_length):
        for z in range(0, house_width):
            for y in range(1):
                if x == 2 or x == house_length - 3 or z == 2 or z == house_width - 3:
                    ED.placeBlock((house_x + x, house_y + house_height + y + 2, house_z + z), Block(material))
                else:
                    ED.placeBlock((house_x + x, house_y + house_height + y + 2, house_z + z), Block("minecraft:air"))

    for x in range(0, house_length):
        for z in range(0, house_width):
            for y in range(1):
                if x == 3 or x == house_length - 4 or z == 3 or z == house_width - 4:
                    ED.placeBlock((house_x + x, house_y + house_height + y + 3, house_z + z), Block(material))
                else:
                    ED.placeBlock((house_x + x, house_y + house_height + y + 3, house_z + z), Block("minecraft:air"))

    for x in range(4, house_length - 4):
        for z in range(4, house_width - 4):
            for y in range(1):
                ED.placeBlock((house_x + x, house_y + house_height + y + 3, house_z + z), Block("minecraft:glass"))
    print("roof built with {}".format(material))
