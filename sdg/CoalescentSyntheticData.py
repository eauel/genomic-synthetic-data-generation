import msprime
import zarr
from numcodecs import Blosc
from etaprogress.progress import ProgressBar
import sys

ZARR_PATH = './output.zarr'
DS_VARIANTS_PER_CHUNK = 16384
DS_SAMPLES_PER_CHUNK = 16384
DS_PLOIDY_PER_CHUNK = None

MEM_VARIANTS_PER_CHUNK = 1
MEM_SAMPLES_PER_CHUNK = None
MEM_PLOIDY_PER_CHUNK = None


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
    z_chunks = (DS_VARIANTS_PER_CHUNK, DS_SAMPLES_PER_CHUNK, DS_PLOIDY_PER_CHUNK)
    z = root.empty('calldata/GT', shape=z_shape,
                   chunks=z_chunks,
                   dtype='i1',
                   compressor=compressor)

    num_variants = z.shape[0]
    num_samples = z.shape[1]
    num_ploidy = z.shape[2]

    print('Num Samples: {}'.format(num_samples))
    print('Num Variants: {}'.format(num_variants))
    print('Ploidy: {}'.format(num_ploidy))
    print("Variation rate: {}".format(num_variants / num_bases))

    bar = ProgressBar(tree_sequence.get_num_mutations(), max_width=80)
    print("Pulling variant data...")
    z_temp_chunks = (MEM_VARIANTS_PER_CHUNK, MEM_SAMPLES_PER_CHUNK, MEM_PLOIDY_PER_CHUNK)
    z_temp = zarr.zeros(shape=z_shape, compressor=compressor, chunks=z_temp_chunks)
    variant_counter = 0
    for variant in tree_sequence.variants():
        bar.numerator = variant_counter
        print(bar, end='\r')
        sys.stdout.flush()

        var = variant.genotypes.reshape((num_samples, ploidy))
        z_temp[variant.index, :, :] = var
        variant_counter += 1

    # Update the Zarr directory store with the data in memory
    z[:, :, :] = z_temp

    print('Done.\n')
    print(z.info)


if __name__ == '__main__':
    generate_coalescent_synthetic_data(num_samples=1000,
                                       num_bases=3e5,
                                       Ne=1e4,
                                       mu=3.5e-9,
                                       rrate=0,
                                       ploidy=2,
                                       seed=57)
