import msprime
import numpy as np
import dask.array as da
from sdg.DaskUtils import *

ZARR_PATH = './output.zarr'


def generate_coalescent_synthetic_data(num_samples=1000, num_bases=1e7, Ne=1e4, mu=3.5e-9, rrate=1e-8,
                                       ploidy=2, seed=57):
    """
        Function credits: Nick Harding
        Reference URL: https://hardingnj.github.io/2017/08/23/power-of-correct-tools.html

    tree_sequence = msprime.simulate(population_configurations=pop_cfg,
                                     demographic_events=dem_hist,
                                     length=num_bases,
                                     recombination_rate=rrate,
                                     mutation_rate=mu,
                                     random_seed=seed)
    """
    tree_sequence = msprime.simulate(
        sample_size=num_samples * ploidy, Ne=Ne, length=num_bases, recombination_rate=rrate, mutation_rate=mu,
        random_seed=seed)

    # Print the number of mutations in tree sequence
    print("Simulated ", tree_sequence.get_num_mutations(), "mutations")

    V = None

    print("Pulling variant data...")

    for variant in tree_sequence.variants():
        var = np.zeros(shape=(1, variant.genotypes.shape[0]))
        var[0, :] = variant.genotypes
        if V is not None:
            V = da.concatenate([V, var], axis=0)
        else:
            V = var
            # V = V.reshape((-1, 1))

    del tree_sequence

    print("  - Done.")

    # Reshape data
    newshape = (V.shape[0], -1, ploidy)
    V = V.reshape(newshape)

    num_variants = V.shape[0]
    num_samples = V.shape[1]
    num_ploidy = V.shape[2]

    print('Num Samples: {}'.format(num_samples))
    print('Num Variants: {}'.format(num_variants))
    print('Ploidy: {}'.format(num_ploidy))
    print("Variation rate: {}".format(num_variants / num_bases))

    # Save the genotype data to zarr store
    print('Saving data to zarr directory')
    gt_output = V.rechunk(chunks=(65536, 64, ploidy))
    da.to_zarr(arr=gt_output, url=ZARR_PATH, component='calldata/GT', overwrite=True)

    print('done')


if __name__ == '__main__':
    connect_dask(address='127.0.0.1', port=8786)

    generate_coalescent_synthetic_data(num_samples=2548,
                                       num_bases=50e6,
                                       Ne=1e4,
                                       mu=3.5e-9,
                                       rrate=1e-8,
                                       ploidy=2,
                                       seed=57)
