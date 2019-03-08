import dask.array as da
import numpy as np
from distributed import Client
import allel
import zarr


def connect_dask(address, port):
    # Connect to Dask scheduler
    print('[Dask Utils] Connecting to Dask scheduler.')
    client = Client('{}:{}'.format(address, port))
    return client


def generate_random_genotype_data(block: np.ndarray, ones_rate: float):
    x, y, z = block.shape
    num_elements = x * y * z
    # Reshape block data to 1-dim
    block = block.reshape((num_elements,))

    num_ones = int(min(ones_rate * num_elements, num_elements))  # Ensure ones_rate doesn't exceed 1

    # Add ones to array at randomly-selected locations
    idx = np.random.choice(block.shape[0], num_ones, replace=False)
    block[idx] = 1

    # Reshape the array to original shape
    block = block.reshape((x, y, z))

    return block


def generate_random_synthetic_data(num_variants, num_samples, heterozygosity_rate, ploidy=2):
    # Create an empty array of zeros
    genetic_data: da.Array = da.zeros(shape=(num_variants, num_samples, ploidy),
                                      dtype=np.int8,
                                      chunks=(2048, 2048, 2))

    # Fill in random ones based on heterozygosity rate
    genetic_data = da.map_blocks(generate_random_genotype_data,
                                 genetic_data,
                                 dtype=np.int8,
                                 ones_rate=heterozygosity_rate)

    return genetic_data


def open_synthetic_data(zarr_path):
    store = zarr.DirectoryStore(zarr_path)
    callset = zarr.Group(store=store, read_only=True)
    return callset


def get_genotype_data(callset):
    genotype_ref_name = ''

    # Ensure 'calldata' is within the callset
    if 'calldata' in callset:
        # Try to find either GT or genotype in calldata
        if 'GT' in callset['calldata']:
            genotype_ref_name = 'GT'
        elif 'genotype' in callset['calldata']:
            genotype_ref_name = 'genotype'
        else:
            return None
    else:
        return None

    gtz = callset['calldata'][genotype_ref_name]

    return allel.GenotypeDaskArray(gtz)


ZARR_PATH = './output.zarr'

if __name__ == '__main__':
    connect_dask(address='127.0.0.1', port=8786)

    genotype_data = generate_random_synthetic_data(num_variants=2000000,
                                                   num_samples=100000,
                                                   heterozygosity_rate=0.001,
                                                   ploidy=2)

    # Save the genotype data to zarr store
    da.to_zarr(arr=genotype_data, url=ZARR_PATH, component='calldata/GT', overwrite=True)

    # gda = allel.GenotypeDaskArray(genetic_data)
    # print(gda.count_alleles())

    callset = open_synthetic_data(ZARR_PATH)

    gt = get_genotype_data(callset)
    print(gt)
