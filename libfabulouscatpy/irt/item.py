###############################################################################
#
#                           COPYRIGHT NOTICE
#                  Mark O. Hatfield Clinical Research Center
#                       National Institutes of Health
#            United States Department of Health and Human Services
#
# This software was developed and is owned by the National Institutes of
# Health Clinical Center (NIHCC), an agency of the United States Department
# of Health and Human Services, which is making the software available to the
# public for any commercial or non-commercial purpose under the following
# open-source BSD license.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# (1) Redistributions of source code must retain this copyright
# notice, this list of conditions and the following disclaimer.
# 
# (2) Redistributions in binary form must reproduce this copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
# 
# (3) Neither the names of the National Institutes of Health Clinical
# Center, the National Institutes of Health, the U.S. Department of
# Health and Human Services, nor the names of any of the software
# developers may be used to endorse or promote products derived from
# this software without specific prior written permission.
# 
# (4) Please acknowledge NIHCC as the source of this software by including
# the phrase "Courtesy of the U.S. National Institutes of Health Clinical
# Center"or "Source: U.S. National Institutes of Health Clinical Center."
# 
# THIS SOFTWARE IS PROVIDED BY THE U.S. GOVERNMENT AND CONTRIBUTORS "AS
# IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
# TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED.
# 
# You are under no obligation whatsoever to provide any bug fixes,
# patches, or upgrades to the features, functionality or performance of
# the source code ("Enhancements") to anyone; however, if you choose to
# make your Enhancements available either publicly, or directly to
# the National Institutes of Health Clinical Center, without imposing a
# separate written license agreement for such Enhancements, then you hereby
# grant the following license: a non-exclusive, royalty-free perpetual license
# to install, use, modify, prepare derivative works, incorporate into
# other computer software, distribute, and sublicense such Enhancements or
# derivative works thereof, in binary and source code form.
#
###############################################################################


import json
from pathlib import Path


class ItemDatabase(object):
    """
    Load items recursively from the path
    """

    def __init__(self, path):

        items = []
        for elem in Path(path).rglob("*.json"):
            items += [json.load(open(elem, "r"))]
        self.items = items
        self.pos = 0

    def __iter__(self):
        self.pos = 0
        return self

    def __next__(self):
        self.pos += 1
        if self.pos > len(self.items):
            raise StopIteration
        return self.items[self.pos - 1]




class ScaleDatabase(object):
    def __init__(self, filename: str) -> None:
        try:
            self.scales = json.load(open(filename, "r"))
        except FileNotFoundError:
            return None
        self.pos = 0
        self.scale_keys = list(self.scales.keys())

    def __iter__(self):
        self.pos = 0
        return self

    def __next__(self):
        self.pos += 1
        if self.pos > len(self.scale_keys):
            raise StopIteration
        scale = self.scale_keys[self.pos - 1]
        return scale, self.scales[scale]

    def exclude_scale(self, scale: str) -> None:
        try:
            del self.scales[scale]
        except KeyError:
            pass

