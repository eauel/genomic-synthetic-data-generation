import dask.array as da
import zarr

from LoadData import *


class DataConcatenator:
    def __init__(self):
        pass

    def get_combined_genotype_array(self, zarr_paths, axis=0):
        if not isinstance(zarr_paths, list):
            raise TypeError('zarr_paths should be a list of zarr paths to combine into a single array.')

        gt_list = []

        for zarr_path in zarr_paths:
            callset = open_zarr_dataset(zarr_path)
            gt = get_genotype_data(callset)

            gt_list.append(gt)

        combined_gt = da.concatenate(gt_list, axis=axis)
        combined_gt = allel.GenotypeDaskArray(combined_gt)

        return combined_gt


if __name__ == '__main__':
    data_concat = DataConcatenator()

    zarr_locations = ['./output1.zarr',
                      './output2.zarr']

    combined_gt = data_concat.get_czombined_genotype_array(zarr_locations)
