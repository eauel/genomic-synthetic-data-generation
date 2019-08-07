import msprime
import zarr
from numcodecs import Blosc
from etaprogress.progress import ProgressBar
import sys

ZARR_PATH = './output.zarr'
VARIANTS_PER_CHUNK = 16384
SAMPLES_PER_CHUNK = 16384
PLOIDY_PER_CHUNK = None

TEMP_VARIANTS_PER_CHUNK = 1
TEMP_SAMPLES_PER_CHUNK = None
TEMP_PLOIDY_PER_CHUNK = None


def generate_coalescent_synthetic_data(num_samples=1000, num_bases=1e7, Ne=1e4, mu=3.5e-9, rrate=1e-8,
                                       ploidy=2, seed=57):
    """
        Function credits: Nick Harding
        Reference URL: https://hardingnj.github.io/2017/08/23/power-of-correct-tools.html
    """
    tree_sequence = msprime.simulate(
        sample_size=num_samples * ploidy,
        Ne=Ne,
        length=num_bases,
        recombination_rate=rrate,
        mutation_rate=mu,
        random_seed=seed,
        model="dtwf")

    # Print the number of mutations in tree sequence
    print("Simulated ", tree_sequence.get_num_mutations(), "mutations")

    print("Creating Zarr data store root")
    store = zarr.DirectoryStore(ZARR_PATH)
    root = zarr.group(store=store, overwrite=True)

    print('Creating Zarr Array')
    compressor = Blosc(cname='zstd', clevel=1, shuffle=Blosc.AUTOSHUFFLE)
    z_shape = (tree_sequence.get_num_mutations(), num_samples, ploidy)
    z_chunks = (VARIANTS_PER_CHUNK, SAMPLES_PER_CHUNK, PLOIDY_PER_CHUNK)
    z = root.empty('calldata/GT', shape=z_shape,
                   chunks=z_chunks,
                   dtype='i1',
                   compressor=compressor)

    print('Creating temporary Zarr Array for holding data')
    temp_chunks = (TEMP_VARIANTS_PER_CHUNK, TEMP_SAMPLES_PER_CHUNK, TEMP_PLOIDY_PER_CHUNK)
    temp_store = zarr.TempStore(dir='./')
    temp_root = zarr.group(store=temp_store, overwrite=True)
    temp_z = temp_root.empty('calldata/GT', shape=z_shape,
                             chunks=temp_chunks,
                             dtype='i1',
                             compressor=None)

    num_variants = z.shape[0]
    num_samples = z.shape[1]
    num_ploidy = z.shape[2]

    print('Num Samples: {}'.format(num_samples))
    print('Num Variants: {}'.format(num_variants))
    print('Ploidy: {}'.format(num_ploidy))
    print("Variation rate: {}".format(num_variants / num_bases))

    bar = ProgressBar(tree_sequence.get_num_mutations(), max_width=80)
    print("Pulling variant data...")
    variant_counter = 0
    for variant in tree_sequence.variants():
        bar.numerator = variant_counter
        print(bar, end='\r')
        sys.stdout.flush()

        var = variant.genotypes.reshape((num_samples, ploidy))
        temp_z[variant.index, :, :] = var
        variant_counter += 1

    # Store data in final data store
    z[:, :, :] = temp_z

    print('Done.\n')
    print(z.info)


if __name__ == '__main__':
    generate_coalescent_synthetic_data(num_samples=100000,
                                       num_bases=3e9,
                                       Ne=1e4,
                                       mu=3.5e-9,
                                       rrate=0,
                                       ploidy=2,
                                       seed=57)
