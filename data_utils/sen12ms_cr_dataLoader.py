"""
    Generic data loading routines for the SEN12MS-CR dataset of corresponding Sentinel 1,
    Sentinel 2 and cloudy Sentinel 2 data.

    The SEN12MS-CR class is meant to provide a set of helper routines for loading individual
    image patches as well as triplets of patches from the dataset. These routines can easily
    be wrapped or extended for use with many deep learning frameworks or as standalone helper 
    methods. For an example use case please see the "main" routine at the end of this file.

    NOTE: Some folder/file existence and validity checks are implemented but it is 
          by no means complete.

    Authors: Patrick Ebel (patrick.ebel@tum.de), Lloyd Hughes (lloyd.hughes@tum.de),
    based on the exemplary data loader code of https://mediatum.ub.tum.de/1474000, with minimal modifications applied.
"""

import os
import rasterio

import numpy as np

from enum import Enum
from glob import glob


class S1Bands(Enum):
    VV = 1
    VH = 2
    ALL = [VV, VH]
    NONE = []


class S2Bands(Enum):
    B01 = aerosol = 1
    B02 = blue = 2
    B03 = green = 3
    B04 = red = 4
    B05 = re1 = 5
    B06 = re2 = 6
    B07 = re3 = 7
    B08 = nir1 = 8
    B08A = nir2 = 9
    B09 = vapor = 10
    B10 = cirrus = 11
    B11 = swir1 = 12
    B12 = swir2 = 13
    ALL = [B01, B02, B03, B04, B05, B06, B07, B08, B08A, B09, B10, B11, B12]
    RGB = [B04, B03, B02]
    NONE = []


class Seasons(Enum):
    SPRING = "ROIs1158_spring"
    SUMMER = "ROIs1868_summer"
    FALL = "ROIs1970_fall"
    WINTER = "ROIs2017_winter"
    ALL = [SPRING, SUMMER, FALL, WINTER]


class Sensor(Enum):
    s1 = "s1"
    s2 = "s2"
    s2cloudy = "s2cloudy"

# Note: The order in which you request the bands is the same order they will be returned in.


class SEN12MSCRDataset:
    def __init__(self, base_dir):
        self.base_dir = base_dir

        if not os.path.exists(self.base_dir):
            raise Exception(
                "The specified base_dir for SEN12MS-CR dataset does not exist")

    """
        Returns a list of scene ids for a specific season.
    """

    def get_scene_ids(self, season):
        season = Seasons(season).value
        path = os.path.join(self.base_dir, season)
        paths = [p for p in os.listdir(self.base_dir) if os.path.isdir(os.path.join(self.base_dir, p))]
        self.paths = paths
        # put s2, s2_cloudy in season or change the script

        """
        if not os.path.exists(path):
            print("path:", path)
            print(os.listdir(self.base_dir))
            raise NameError("Could not find season {} in base directory {}".format(
                season, self.base_dir))"""

        # add all dirs except "s2_cloudy" (which messes with subsequent string splits)
        #scene_list = [os.path.basename(s)
        #              for s in glob(os.path.join(path, "*")) if "s2_cloudy" not in s]
        scene_list = [os.path.join(self.base_dir, s) for s in paths if "s2_cloudy" not in s and ("s1" in s or "s2" in s)]
        scene_list = [int(s_id.split("_")[1]) for s in scene_list for s_id in os.listdir(s)]
        #scene_list = [int(s.split("_")[1]) for s in scene_list]
        return set(scene_list)

    """
        Returns a list of patch ids for a specific scene within a specific season
    """

    def get_patch_ids(self, season, scene_id):
        season = Seasons(season).value
        path = os.path.join(self.base_dir, season, f"s2_{scene_id}") # expect ROI_SEASON\s2_scene_id. Do move operations. From cvommand line or manually.
        """
        if not os.path.exists(path):
            raise NameError(
                "Could not find scene {} within season {}".format(scene_id, season))"""

        patch_ids = [os.path.splitext(os.path.basename(p))[0]
                     for p in glob(os.path.join(path, "*"))]
        patch_ids = [int(p.rsplit("_", 1)[1].split("p")[1]) for p in patch_ids]
        breakpoint()

        return patch_ids

    """
        Return a dict of scene ids and their corresponding patch ids.
        key => scene_ids, value => list of patch_ids
    """

    def get_season_ids(self, season):
        season = Seasons(season).value
        ids = {}
        scene_ids = self.get_scene_ids(season)

        for sid in scene_ids:
            ids[sid] = self.get_patch_ids(season, sid)

        return ids

    """
        Returns raster data and image bounds for the defined bands of a specific patch
        This method only loads a sinlge patch from a single sensor as defined by the bands specified
    """

    def get_patch(self, season, scene_id, patch_id, bands):
        season = Seasons(season).value
        sensor = None

        if isinstance(bands, (list, tuple)):
            b = bands[0]
        else:
            b = bands
        
        if isinstance(b, S1Bands):
            sensor = Sensor.s1.value
            bandEnum = S1Bands
        elif isinstance(b, S2Bands):
            sensor = Sensor.s2.value
            bandEnum = S2Bands
        else:
            raise Exception("Invalid bands specified")

        if isinstance(bands, (list, tuple)):
            bands = [b.value for b in bands]
        else:
            bands = bands.value

        scene = "{}_{}".format(sensor, scene_id)
        filename = "{}_{}_p{}.tif".format(season, scene, patch_id)
        patch_path = os.path.join(self.base_dir, season, scene, filename)

        with rasterio.open(patch_path) as patch:
            data = patch.read(bands)
            bounds = patch.bounds

        if len(data.shape) == 2:
            data = np.expand_dims(data, axis=0)

        return data, bounds

    """
        Returns a triplet of patches. S1, S2 and cloudy S2 as well as the geo-bounds of the patch
    """

    def get_s1s2s2cloudy_triplet(self, season, scene_id, patch_id, s1_bands=S1Bands.ALL, s2_bands=S2Bands.ALL, s2cloudy_bands=S2Bands.ALL):
        s1, bounds = self.get_patch(season, scene_id, patch_id, s1_bands)
        s2, _ = self.get_patch(season, scene_id, patch_id, s2_bands)
        s2cloudy, _ = self.get_patch(season, scene_id, patch_id, s2cloudy_bands)

        return s1, s2, s2cloudy, bounds

    """
        Returns a triplet of numpy arrays with dimensions D, B, W, H where D is the number of patches specified
        using scene_ids and patch_ids and B is the number of bands for S1, S2 or cloudy S2
    """

    def get_triplets(self, season, scene_ids=None, patch_ids=None, s1_bands=S1Bands.ALL, s2_bands=S2Bands.ALL, s2cloudy_bands=S2Bands.ALL):
        season = Seasons(season)
        scene_list = []
        patch_list = []
        bounds = []
        s1_data = []
        s2_data = []
        s2cloudy_data = []

        # This is due to the fact that not all patch ids are available in all scenes
        # And not all scenes exist in all seasons
        if isinstance(scene_ids, list) and isinstance(patch_ids, list):
            raise Exception("Only scene_ids or patch_ids can be a list, not both.")

        if scene_ids is None:
            scene_list = self.get_scene_ids(season)
        else:
            try:
                scene_list.extend(scene_ids)
            except TypeError:
                scene_list.append(scene_ids)

        if patch_ids is not None:
            try:
                patch_list.extend(patch_ids)
            except TypeError:
                patch_list.append(patch_ids)

        for sid in scene_list:
            if patch_ids is None:
                patch_list = self.get_patch_ids(season, sid)

            for pid in patch_list:
                s1, s2, s2cloudy, bound = self.get_s1s2s2cloudy_triplet(
                    season, sid, pid, s1_bands, s2_bands, s2cloudy_bands)
                s1_data.append(s1)
                s2_data.append(s2)
                s2cloudy_data.append(s2cloudy)
                bounds.append(bound)

        return np.stack(s1_data, axis=0), np.stack(s2_data, axis=0), np.stack(s2cloudy_data, axis=0), bounds

def test():
    path = "H:\Fulvio\data\SEN12MS_CR"
    breakpoint()

if __name__ == "__main__":
    import time
    # Load the dataset specifying the base directory
    sen12mscr = SEN12MSCRDataset("H:\Fulvio\data\SEN12MS_CR")

    spring_ids = sen12mscr.get_season_ids(Seasons.SPRING)
    cnt_patches = sum([len(pids) for pids in spring_ids.values()])
    print("Spring: {} scenes with a total of {} patches".format(
        len(spring_ids), cnt_patches))

    start = time.time()
    # Load the RGB bands of the first S2 patch in scene 8
    SCENE_ID = 8
    breakpoint()
    s2_rgb_patch, bounds = sen12mscr.get_patch(Seasons.SPRING, SCENE_ID, 
                                            spring_ids[SCENE_ID][0], bands=S2Bands.RGB)
    print("Time Taken {}s".format(time.time() - start))
                                            
    print("S2 RGB: {} Bounds: {}".format(s2_rgb_patch.shape, bounds))

    print("\n")

    # Load a triplet of patches from the first three scenes of Spring - all S1 bands, NDVI S2 bands, and NDVI S2 cloudy bands
    i = 0
    start = time.time()
    for scene_id, patch_ids in spring_ids.items():
        if i >= 3:
            break

        s1, s2, s2cloudy, bounds = sen12mscr.get_s1s2s2cloudy_triplet(Seasons.SPRING, scene_id, patch_ids[0], s1_bands=S1Bands.ALL,
                                                        s2_bands=[S2Bands.red, S2Bands.nir1], s2cloudy_bands=[S2Bands.red, S2Bands.nir1])
        print(
            f"Scene: {scene_id}, S1: {s1.shape}, S2: {s2.shape}, cloudy S2: {s2cloudy.shape}, Bounds: {bounds}")
        i += 1

    print("Time Taken {}s".format(time.time() - start))
    print("\n")

    start = time.time()
    # Load all bands of all patches in a specified scene (scene 106)
    s1, s2, s2cloudy, _ = sen12mscr.get_triplets(Seasons.SPRING, 106, s1_bands=S1Bands.ALL, 
                                        s2_bands=S2Bands.ALL, s2cloudy_bands=S2Bands.ALL)

    print(f"Scene: 106, S1: {s1.shape}, S2: {s2.shape}, cloudy S2: {s2cloudy.shape}")
    print("Time Taken {}s".format(time.time() - start))
