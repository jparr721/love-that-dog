#!/usr/bin/env python3

from PIL import Image


def rotate_image(img_path, output_path, deg):
    """
    Rotates an image a given number of degrees

    Parameters
    ----------
    img_path : str
        The absolute path of the image
    output_path : str
        The place to save the image
    deg : int
        The degrees of rotation
    """
    img = Image.open(img_path)
    rotated_img = img.rorate(deg)

    rotated_img.save(output_path + f"/rotated_{deg}.jpg")


def adjust_brightness(img_path, output_path, pct):
    """
    Adjusts the brightness up or down on an image and saves it
    to the output path

    Parameters
    ----------
    img_path : str
        The absolute path of the image
    output_path : str
        The place to save the image
    pct : float
        How much to adjust the brightness by
    """
    img = Image.open(img_path)
    adjusted = img.point(lambda p: p * pct)

    adjusted.save(output_path + f"/adjusted_{pct}.jpg")
