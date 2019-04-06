from distributed import Client

from sdg.RandomSyntheticData import *


def connect_dask(address, port):
    # Connect to Dask scheduler
    print('[Dask Utils] Connecting to Dask scheduler.')
    client = Client('{}:{}'.format(address, port))
    return client


ZARR_PATH = './output.zarr'


def main():
    connect_dask(address='127.0.0.1', port=8786)

    genotype_data = generate_random_synthetic_data(num_variants=73257633,
                                                   num_samples=100000,
                                                   heterozygosity_rate=0.001,
                                                   ploidy=2)

    # Save the genotype data to zarr store
    da.to_zarr(arr=genotype_data, url=ZARR_PATH, component='calldata/GT', overwrite=True)


if __name__ == '__main__':
    main()
