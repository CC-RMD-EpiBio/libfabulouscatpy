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

from abc import ABC

from libfabulouscatpy.irt.item import ItemDatabase, ScaleDatabase


class FactorizedIRTModel(ABC):
    def __init__(self,
            itemdb: ItemDatabase,
            scaledb: ScaleDatabase,
            ):
        self.itemdb = itemdb
        self.scaledb = scaledb    

class IRTModel(object):
    def __init__(self,
            itemdb: ItemDatabase,
            scaledb: ScaleDatabase,
            ):
        self.itemdb = itemdb
        self.scaledb = scaledb    
        
class MultivariateIRTModel(ABC):
    def __init__(self,
            itemdb: ItemDatabase,
            scaledb: ScaleDatabase,
            ):
        self.itemdb = itemdb
        self.scaledb = scaledb    
        
class EngineItemArrays(object):
    """
    A class to keep track of the parameters associated to a particular engine.
    """

    def __init__(self, slope=[], calibration=[]):
        self.slope = slope
        self.calibration = calibration
        self.category = map(lambda x: len(x) + 1, self.calibration)


class Calibrations(object):
    def __init__(self, num, category, calibration):
        """
        Parameters: num: int
                        The ID associated to the particular calibration.
                    category: int
                        The category associated to the calibration.
                    calibration: float
                        The actual calibration value.
        """

        self.num = num
        self.category = category
        self.calibration = calibration
        # self.item is a placeholder to later be populated by an instance of
        # ItemsBase().
        self.item = None
        self.to_string = "Calibrations[id = %s]" % num

    def __eq__(self, other=None):
        """
        Parameters: other: instance of Calibrations().
        Returns:    w:  bool
                    True if equal, False otherwise.
        """

        if other is None:
            return False

        else:
            return self.num == other.num
        
  