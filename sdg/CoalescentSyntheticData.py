import msprime
import allel
import numpy as np
import dask.array as da

ZARR_PATH = './output.zarr'


def apply_variant_data_block(block: np.ndarray,
                             block_info=None,
                             variant_data: np.ndarray = None,
                             locations: np.ndarray = None):
    # Determine current location in the output genotype array
    output_idx = block_info[0]['chunk-location'][0]

    if output_idx in locations:
        var_idx = np.where(locations == output_idx)[0][0]
        new_block = variant_data[var_idx, :, :].reshape(block.shape)
        return new_block
    else:
        return block


def generate_coalescent_synthetic_data(num_bases=1e7, mu=3.5e-9, rrate=1e-8, sample_size=1000, Ne=1, ploidy=2,
                                       seed=57):
    """
        Function credits: Nick Harding
        Reference URL: https://hardingnj.github.io/2017/08/23/power-of-correct-tools.html
    """
    tree_sequence = msprime.simulate(length=num_bases,
                                     recombination_rate=rrate,
                                     mutation_rate=mu,
                                     sample_size=sample_size * ploidy,  # Need to mult. by ploidy due to array folding
                                     Ne=Ne,
                                     random_seed=seed)

    # Print the number of mutations in tree sequence
    print("Simulated ", tree_sequence.get_num_mutations(), "mutations")

    V = np.zeros(shape=(tree_sequence.get_num_mutations(), tree_sequence.get_sample_size()), dtype=np.int8)

    for variant in tree_sequence.variants():
        V[variant.index] = variant.genotypes

    ht = allel.HaplotypeArray(V)
    gt = ht.to_genotypes(ploidy=ploidy)

    print('Num Allele Calls: {}'.format(gt.n_allele_calls))
    print('Num Calls: {}'.format(gt.n_calls))
    print('Num Samples: {}'.format(gt.n_samples))
    print('Num Variants: {}'.format(gt.n_variants))

    gt = gt.values  # Get underlying NumPy array

    # Create the full output array
    gt_output = da.zeros(shape=(num_bases, sample_size, ploidy), dtype=np.int8)

    # Find random locations to place the generated variants
    vidx = np.random.choice(gt_output.shape[0], gt.shape[0], replace=False)

    # Rechunk output data to perform a mapping across entire rows
    gt_output = gt_output.rechunk((1, -1, -1))

    gt_output = da.map_blocks(apply_variant_data_block, gt_output, dtype=np.int8, variant_data=gt, locations=vidx)

    # Save the genotype data to zarr store
    print('Saving data to zarr directory')
    gt_output = gt_output.rechunk((65536, 64, ploidy))
    da.to_zarr(arr=gt_output, url=ZARR_PATH, component='calldata/GT', overwrite=True)

    print('done')


if __name__ == '__main__':
    """
    generate_coalescent_synthetic_data(num_bases=10000,
                                       mu=3.5e-9,
                                       rrate=1e-8,
                                       sample_size=2548,
                                       Ne=1000,  # Effective diploid population size
                                       ploidy=2,
                                       seed=57)
    """
    generate_coalescent_synthetic_data(num_bases=10000,
                                       mu=0.000001,
                                       rrate=1e-8,
                                       sample_size=2548,
                                       Ne=1000,  # Effective diploid population size
                                       ploidy=2,
                                       seed=57)

