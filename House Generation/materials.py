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

def material_selection(usage = None):
    if usage == "wall" or usage == "foundation":
        materials = ["minecraft:stone", "minecraft:bricks", "minecraft:cobblestone", "minecraft:stone_bricks",
                     "minecraft:smooth_quartz"]

    elif usage == "roof":
        materials = ["minecraft:spruce_wood", "minecraft:jungle_wood", "minecraft:acacia_wood",
                     "minecraft:dark_oak_wood", "minecraft:oak_wood", "minecraft:birch_wood"]

    elif usage == "window":
        materials = ['"minecraft:glass"', '"minecraft:glass_pane"']

    elif usage == "door":
        materials = ["minecraft:dark_oak_door", "minecraft:spruce_door", "minecraft:oak_door",
                     "minecraft:birch_door", "minecraft:acacia_door", "minecraft:jungle_door"]

    elif usage == "carpet":
        materials = ["minecraft:lime_carpet", "minecraft:pink_carpet", "minecraft:gray_carpet", "minecraft:cyan_carpet",
                     "minecraft:white_carpet", "minecraft:brown_carpet","minecraft:black_carpet",
                     "minecraft:moss_carpet"]

    elif usage == "flower":
        materials = ["minecraft:potted_poppy", "minecraft:potted_blue_orchid", "minecraft:potted_allium",
                     "minecraft:potted_azure_bluet", "minecraft:potted_red_tulip", "minecraft:potted_orange_tulip",
                     "minecraft:potted_white_tulip", "minecraft:potted_pink_tulip", "minecraft:potted_oxeye_daisy",
                     "minecraft:potted_dandelion", "minecraft:potted_oxeye_daisy", "minecraft:potted_dandelion"]

    elif usage == "torch":
        materials = ["minecraft:torch", "minecraft:soul_torch", "minecraft:redstone_torch"]

    elif usage == "light":
        materials = ["minecraft:lantern", "minecraft:soul_lantern"]

    elif usage == "bed":
        materials = ["minecraft:red_bed", "minecraft:gray_bed", "minecraft:light_blue_bed",
                     "minecraft:light_gray_bed", "minecraft:lime_bed", "minecraft:orange_bed"]

    material = np.random.choice(materials)
    return material

    
