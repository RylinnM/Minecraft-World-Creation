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


def buildDoor(house_x, house_y, house_z, house_height, house_width, house_length):
    print("building door...")
    material = material_selection("door")
    house_x_center = int(house_x + house_length / 2)
    house_z_center = int(house_z + house_width / 2)
    door_height = 2
    x = house_x_center
    z = house_z
    for y in range(house_y, house_y + door_height):
        ED.placeBlock((x, y, z), Block("minecraft:air"))
    ED.placeBlock((x, house_y-1, z), Block("minecraft:stone"))
    ED.placeBlock((x, house_y, z), Block(material, {"hinge": "left", "half": "lower"}))
    ED.placeBlock((x, house_y + 1, z), Block(material, {"hinge": "left", "half": "upper"}))
    # place a stair block to make the door look nicer and keep the facing direction to the door
    ED.placeBlock((x, house_y, z+1), Block("minecraft:air"))
    ED.placeBlock((x, house_y, z+2), Block("minecraft:air"))
    ED.placeBlock((x, house_y-1, z+1), Block("minecraft:stone"))
    ED.placeBlock((x, house_y+1, z+1), Block("minecraft:air"))
    ED.placeBlock((x, house_y+2, z+2), Block("minecraft:air"))
    ED.placeBlock((x, house_y, z + 2), Block("minecraft:air"))
    ED.placeBlock((x, house_y-1, z-1), Block("minecraft:quartz_stairs", {"facing": "south"}))
    print("door built with: " + material)


def buildWindows(house_x, house_y, house_z, house_height, house_width, house_length):
    print("building windows...")
    window_y = int(house_y + house_height / 2) - 1
    window_height = 2
    window_width = 3

    # build windows on the long side
    window_x1 = int(house_x + 1/4 * house_length)
    window_x2 = int(house_x + 3/4 * house_length)
    window_z1 = house_z
    window_z2 = house_z + house_width - 1
    for x in range(int(window_x1- window_width /2), int(window_x1 + window_width/2)):
        for y in range(window_y, window_y + window_height):
            z = window_z1
            ED.placeBlock((x+1, y, z), Block("minecraft:glass"))
            z = window_z2
            ED.placeBlock((x+1, y, z), Block("minecraft:glass"))

    for x in range(int(window_x2 - window_width/2), int(window_x2 + window_width/2)):
        for y in range(window_y, window_y + window_height):
            z = window_z1
            ED.placeBlock((x+1, y, z), Block("minecraft:glass"))
            z = window_z2
            ED.placeBlock((x+1, y, z), Block("minecraft:glass"))

    # build windows on the short side
    window_width = 2
    window_x1 = house_x
    window_x2 = house_x + house_length - 1
    window_z1 = int(house_z + 1/2 * house_width)
    for z in range(int(window_z1- window_width/2), int(window_z1 + window_width/2)):
        for y in range(window_y, window_y + window_height):
            x = window_x1
            ED.placeBlock((x, y, z), Block("minecraft:glass"))
            x = window_x2
            ED.placeBlock((x, y, z), Block("minecraft:glass"))
    print("windows built")


def buildLights(house_x, house_y, house_z, house_height, house_width, house_length):
    print("building external lights...")
    material = material_selection("torch")
    light_y = house_y + house_height + 1
    light_x1 = house_x
    light_x2 = house_x + house_length - 1
    light_z1 = house_z
    light_z2 = house_z + house_width - 1
    ED.placeBlock((light_x1, light_y, light_z1), Block(material))
    ED.placeBlock((light_x1, light_y, light_z2), Block(material))
    ED.placeBlock((light_x2, light_y, light_z1), Block(material))
    ED.placeBlock((light_x2, light_y, light_z2), Block(material))
    print("external lights built with", material)


def clear_prem(house_x, house_y, house_z, house_height, house_width, house_length):
    # clear the blocks directly surrounds the house's door
    print("clearing prem...")
    x = house_x + house_length // 2
    z = house_z - 1
    for y in range(house_y, house_y + 4):
        ED.placeBlock((x, y, z), Block("minecraft:air"))
        ED.placeBlock((x, y, z-1), Block("minecraft:air"))
    print("prem cleared")


def base_rein(house_x, house_y, house_z, house_height, house_width, house_length):

    print("reinforcing base...")
    for x in range(house_x, house_x + house_length + 1):
        for z in range(house_z, house_z + house_width + 1):
            if ED.getBlock((x, house_y - 1, z)) == Block("minecraft:air"):
                ED.placeBlock((x, house_y - 1, z), Block("minecraft:grass_block"))
    print("base reinforced")


def plant_flower(house_x, house_y, house_z, house_height, house_width, house_length):

    print("planting flowers...")
    for x in range(house_x, house_x + house_length + 1):
        for z in range(house_z, house_z + house_width + 1):
            if ED.getBlock((x, house_y - 1, z)) == Block("minecraft:grass_block"):
                ED.placeBlock((x, house_y - 1, z), Block("minecraft:poppy"))
    print("flowers planted")


def build_bell(house_x, house_y, house_z, house_height, house_width, house_length):

    print("building bell...")
    material = "minecraft:bell"
    bell_y = house_y + 2 + 1
    bell_x = house_x + house_length // 2
    bell_z = house_z - 1
    ED.placeBlock((bell_x, bell_y, bell_z), Block(material, {"facing": "south", "attachment": "single_wall"}))
    print("bell built with", material)
