# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
created at: Thu Nov 26 18:11:51 2020
created by: Priyam Tejaswin (tejaswin.p@flipkart.com)

Pygame tests.

The only magic you need to know is called Rotation Matrices.

If you perform multiplication between such a matrix, and a vector, you get that vector rotated.

Armed with this information (i.e. after copying wikipedia's 3D Rotation Matrices), I ended up with this nice thing:
"""

import pygame
from numpy import array
from math import cos, sin


######################
#                    #
#    math section    #
#                    #
######################

X, Y, Z = 0, 1, 2


def rotation_matrix(a, b, g):
    """
    rotation matrix of a, b, g radians around x, y, z axes (respectively)
    """
    sa, ca = sin(a), cos(a)
    sb, cb = sin(b), cos(b)
    sg, cg = sin(g), cos(g)
    return (
        (cb*cg, -cb*sg, sb),
        (ca*sg + sa*sb*cg, ca*cg - sg*sa*sb, -cb*sa),
        (sg*sa - ca*sb*cg, ca*sg*sb + sa*cg, ca*cb)
    )


class Physical:
    def __init__(self, vertices, edges):
        """
        a 3D object that can rotate around the three axes
        :param vertices: a tuple of points (each has 3 coordinates)
        :param edges: a tuple of pairs (each pair is a set containing 2 vertices' indexes)
        """
        self.__vertices = array(vertices)
        self.__edges = tuple(edges)
        self.__rotation = [0, 0, 0]  # radians around each axis

    def rotate(self, axis, t):
        self.__rotation[axis] += t

    @property
    def lines(self):
        location = self.__vertices.dot(rotation_matrix(*self.__rotation))  # an index->location mapping
        return ((location[v1], location[v2]) for v1, v2 in self.__edges)


######################
#                    #
#    gui section     #
#                    #
######################


BLACK, RED = (0, 0, 0), (255, 128, 128)


class Paint:
    def __init__(self, shape, keys_handler):
        self.__shape = shape
        self.__keys_handler = keys_handler
        self.__size = 450, 450
        self.__clock = pygame.time.Clock()
        self.__screen = pygame.display.set_mode(self.__size)
        self.__mainloop()

    def __fit(self, vec):
        """
        ignore the z-element (creating a very cheap projection), and scale x, y to the coordinates of the screen
        """
        # notice that len(self.__size) is 2, hence zip(vec, self.__size) ignores the vector's last coordinate
        return [round(70 * coordinate + frame / 2) for coordinate, frame in zip(vec, self.__size)]

    def __handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()
        self.__keys_handler(pygame.key.get_pressed())

    def __draw_shape(self, thickness=4):
        for start, end in self.__shape.lines:
            pygame.draw.line(self.__screen, RED, self.__fit(start), self.__fit(end), thickness)

    def __mainloop(self):
        while True:
            self.__handle_events()
            self.__screen.fill(BLACK)
            self.__draw_shape()
            pygame.display.flip()
            self.__clock.tick(40)


######################
#                    #
#     main start     #
#                    #
######################


def main():
    from pygame import K_q, K_w, K_a, K_s, K_z, K_x

    cube = Physical(  # 0         1            2            3           4            5            6            7
        vertices=((1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1), (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1)),
        edges=({0, 1}, {0, 2}, {2, 3}, {1, 3},
               {4, 5}, {4, 6}, {6, 7}, {5, 7},
               {0, 4}, {1, 5}, {2, 6}, {3, 7})
    )

    counter_clockwise = 0.05  # radians
    clockwise = -counter_clockwise

    params = {
        K_q: (X, clockwise),
        K_w: (X, counter_clockwise),
        K_a: (Y, clockwise),
        K_s: (Y, counter_clockwise),
        K_z: (Z, clockwise),
        K_x: (Z, counter_clockwise),
    }

    def keys_handler(keys):
        for key in params:
            if keys[key]:
                cube.rotate(*params[key])

    pygame.init()
    pygame.display.set_caption('Control -   q,w : X    a,s : Y    z,x : Z')
    Paint(cube, keys_handler)

if __name__ == '__main__':
    main()

