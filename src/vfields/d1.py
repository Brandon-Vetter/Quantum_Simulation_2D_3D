"""
This file contains 1d methods for E fields.

:Author: Brandon Vetter
:Date: 2/15/26

Property of the University of Idaho
"""

def mirror(line, line_plane):
    for i in range(len(line)):
        line_plane[i] += line[i]
        line_plane[-i] += line[i] 