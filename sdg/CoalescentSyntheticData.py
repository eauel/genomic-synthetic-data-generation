import msprime
import zarr
from numcodecs import Blosc

ZARR_PATH = './output.zarr'


def generate_coalescent_synthetic_data(num_samples=1000, num_bases=1e7, Ne=1e4, mu=3.5e-9, rrate=1e-8,
                                       ploidy=2, seed=57):
    """
        Function credits: Nick Harding
        Reference URL: https://hardingnj.github.io/2017/08/23/power-of-correct-tools.html
    """
    tree_sequence = msprime.simulate(
        sample_size=num_samples * ploidy, Ne=Ne, length=num_bases, recombination_rate=rrate, mutation_rate=mu,
        random_seed=seed)

    # Print the number of mutations in tree sequence
    print("Simulated ", tree_sequence.get_num_mutations(), "mutations")

    print("Creating Zarr data store root")
    store = zarr.DirectoryStore(ZARR_PATH)
    root = zarr.group(store=store, overwrite=True)

    print('Creating Zarr Array')
    compressor = Blosc(cname='zstd', clevel=1, shuffle=Blosc.AUTOSHUFFLE)
    z = root.empty('calldata/GT', shape=(tree_sequence.get_num_mutations(), num_samples, ploidy),
                   chunks=(2048, 2048, ploidy),
                   dtype='i1',
                   compressor=compressor)

    num_variants = z.shape[0]
    num_samples = z.shape[1]
    num_ploidy = z.shape[2]

    print('Num Samples: {}'.format(num_samples))
    print('Num Variants: {}'.format(num_variants))
    print('Ploidy: {}'.format(num_ploidy))
    print("Variation rate: {}".format(num_variants / num_bases))

    print("Pulling variant data...")
    variant_counter = 0
    for variant in tree_sequence.variants():
        print("  Variant {} out of {}".format(variant_counter, tree_sequence.get_num_mutations()))
        var = variant.genotypes.reshape((num_samples, ploidy))
        z[variant.index, :, :] = var
        variant_counter += 1

    print('Done.\n')
    print(z.info)


if __name__ == '__main__':
    generate_coalescent_synthetic_data(num_samples=100000,
                                       num_bases=50e6,
                                       Ne=1e4,
                                       mu=3.5e-9,
                                       rrate=1e-8,
                                       ploidy=2,
                                       seed=57)
