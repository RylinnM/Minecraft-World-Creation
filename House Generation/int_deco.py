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


def applyCarpet(house_x, house_y, house_z, house_height, house_width, house_length):
    print("applying carpet...")
    material = material_selection("carpet")
    for x in range(1, house_length-1):
        for z in range(1, house_width-1):
            for y in range(1):
                ED.placeBlock((house_x + x, house_y + y + 1, house_z + z), Block(material))
    print("carpet applied with {}".format(material))


def addintLights(house_x, house_y, house_z, house_height, house_width, house_length):
    # add lights at the four inner corners of the house
    print("adding internal lights...")
    material = material_selection("light")
    light_x1 = house_x + 3
    light_x2 = house_x + house_length - 4
    light_z1 = house_z + 3
    light_z2 = house_z + house_width - 4
    light_y = house_y + house_height + 2
    ED.placeBlock((light_x1, light_y, light_z1), Block(material, {"hanging": "true"}))
    ED.placeBlock((light_x1, light_y, light_z2), Block(material, {"hanging": "true"}))
    ED.placeBlock((light_x2, light_y, light_z1), Block(material, {"hanging": "true"}))
    ED.placeBlock((light_x2, light_y, light_z2), Block(material, {"hanging": "true"}))
    print("internal lights added with {}".format(material))

def add_int_plants(house_x, house_y, house_z, house_height, house_width, house_length):
    print("adding internal plants...")
    # add plants in the house
    for x in range(1, house_length-1):
        for z in range(1, house_width-1):
            for y in range(1):
                if np.random.random() < 0.2:
                    ED.placeBlock((house_x + x, house_y + y + 1, house_z + z), Block("minecraft:grass_block"))
    print("internal plants added")


def addFurniture(house_x, house_y, house_z, house_height, house_width, house_length):
    print("adding bed...")
    bed_material = material_selection("bed")
    # randomly select a corner
    corner = np.random.choice(["north_west", "north_east", "south_west", "south_east"])
    if corner == "north_west":
        bed_x = house_x + 1
        bed_z = house_z + 2
        table_x = house_x + 1
        table_z = house_z + house_width - 2
        shelf_x = house_x + house_length - 3
        shelf_z = house_z + 2
        ED.placeBlock((table_x, house_y + 1, table_z), Block("minecraft:air"))
        ED.placeBlock((table_x, house_y + 1, table_z), Block("minecraft:crafting_table"))
        ED.placeBlock((table_x, house_y + 1, table_z-1), Block("minecraft:air"))
        ED.placeBlock((table_x, house_y + 1, table_z-1), Block("minecraft:crafting_table"))
        addbookshelf(shelf_x, house_y + 1, shelf_z)
        ED.placeBlock((house_x + house_length -2, house_y + 1, house_z + house_width - 2),
                      Block("minecraft:furnace", {"facing": "west", "lit": "true"}))
        ED.placeBlock((house_x + house_length - 2, house_y + 1, house_z + house_width - 3),
                      Block("minecraft:furnace", {"facing": "west", "lit": "true"}))
    elif corner == "north_east":
        bed_x = house_x + house_length - 2
        bed_z = house_z + 2
        table_x = house_x + house_length - 2
        table_z = house_z + house_width - 2
        shelf_x = house_x + 1
        shelf_z = house_z + 2
        ED.placeBlock((table_x, house_y + 1, table_z), Block("minecraft:air"))
        ED.placeBlock((table_x, house_y + 1, table_z), Block("minecraft:crafting_table"))
        ED.placeBlock((table_x, house_y + 1, table_z-1), Block("minecraft:air"))
        ED.placeBlock((table_x, house_y + 1, table_z-1), Block("minecraft:crafting_table"))
        addbookshelf(shelf_x, house_y + 1, shelf_z)
        ED.placeBlock((house_x + 1, house_y + 1, house_z + house_width - 2),
                      Block("minecraft:furnace", {"facing": "east", "lit": "true"}))
        ED.placeBlock((house_x + 1, house_y + 1, house_z + house_width - 3),
                      Block("minecraft:furnace", {"facing": "east", "lit": "true"}))
    elif corner == "south_west":
        bed_x = house_x + 1
        bed_z = house_z + house_width - 2
        table_x = house_x + 1
        table_z = house_z + 1
        shelf_x = house_x + house_length - 3
        shelf_z = house_z + house_width - 2
        ED.placeBlock((table_x, house_y + 1, table_z), Block("minecraft:air"))
        ED.placeBlock((table_x, house_y + 1, table_z), Block("minecraft:crafting_table"))
        ED.placeBlock((table_x, house_y + 1, table_z+1), Block("minecraft:air"))
        ED.placeBlock((table_x, house_y + 1, table_z+1), Block("minecraft:crafting_table"))
        addbookshelf(shelf_x, house_y + 1, shelf_z)
        ED.placeBlock((house_x + house_length - 2, house_y + 1, house_z + 1),
                      Block("minecraft:furnace", {"facing": "west", "lit": "true"}))
        ED.placeBlock((house_x + house_length - 2, house_y + 1, house_z + 2),
                      Block("minecraft:furnace", {"facing": "west", "lit": "true"}))
    elif corner == "south_east":
        bed_x = house_x + house_length - 2
        bed_z = house_z + house_width - 2
        table_x = house_x + house_length - 2
        table_z = house_z + 1
        shelf_x = house_x + 1
        shelf_z = house_z + house_width - 2
        ED.placeBlock((table_x, house_y + 1, table_z), Block("minecraft:air"))
        ED.placeBlock((table_x, house_y + 1, table_z), Block("minecraft:crafting_table"))
        ED.placeBlock((table_x, house_y + 1, table_z+1), Block("minecraft:air"))
        ED.placeBlock((table_x, house_y + 1, table_z+1), Block("minecraft:crafting_table"))
        addbookshelf(shelf_x, house_y + 1, shelf_z)
        ED.placeBlock((house_x + 1, house_y + 1, house_z + 1),
                      Block("minecraft:furnace", {"facing": "east", "lit": "true"}))
        ED.placeBlock((house_x + 1, house_y + 1, house_z + 2),
                        Block("minecraft:furnace", {"facing": "east", "lit": "true"}))

    ED.placeBlock((bed_x, house_y + 1, bed_z), Block("minecraft:air"))
    ED.placeBlock((bed_x, house_y + 1, bed_z), Block(bed_material, {"facing": "north"}))

    print(f"furniture added with {bed_material}")


def addGarden(house_x, house_y, house_z, house_height, house_width, house_length):
    print("adding garden...")
    material = material_selection("flower")
    x_ref = house_x + house_length // 2
    z_ref = house_z + house_width - 2
    ED.placeBlock((x_ref-1, house_y + 1, z_ref), Block(material))
    #ED.placeBlock((x_ref-1, house_y + 1, z_ref), Block("minecraft:cornflower"))
    ED.placeBlock((x_ref+1, house_y + 1, z_ref), Block(material))
    #ED.placeBlock((x_ref+1, house_y + 1, z_ref), Block("minecraft:cornflower"))
    material = material_selection("flower")
    for x in range(x_ref-1, x_ref+2):
        z = z_ref - 1
        ED.placeBlock((x, house_y + 1, z), Block(material))
    print("garden added with " + material + "s")


def addSign(house_x, house_y, house_z, house_height, house_width, house_length):
    print("adding sign...")
    x_ref = house_x + house_length // 2
    z_ref = house_z + house_width - 2
    ED.placeBlock((x_ref, house_y + 1, z_ref), Block("minecraft:air"))
    ED.placeBlock((x_ref, house_y + 1, z_ref), Block("minecraft:oak_sign", {"rotation": "8"},
                    data="{Text1: '{\"text\": \"Designed by R.Ma\"}'}"))

    print("sign added")

def addbookshelf(x,y,z):
    material = material_selection("flower")
    ED.placeBlock((x, y, z), Block("minecraft:air"))
    ED.placeBlock((x, y, z), Block("minecraft:bookshelf"))
    ED.placeBlock((x + 1, y, z), Block("minecraft:air"))
    ED.placeBlock((x+1, y, z), Block("minecraft:bookshelf"))
    ED.placeBlock((x, y, z-1), Block("minecraft:air"))
    ED.placeBlock((x, y, z-1), Block("minecraft:bookshelf"))
    ED.placeBlock((x+1, y, z-1), Block("minecraft:air"))
    ED.placeBlock((x+1, y, z-1), Block("minecraft:bookshelf"))
    ED.placeBlock((x, y+1, z), Block(material))
    ED.placeBlock((x+1, y+1, z), Block(material))
    ED.placeBlock((x, y+1, z-1), Block(material))
    ED.placeBlock((x+1, y+1, z-1), Block(material))


