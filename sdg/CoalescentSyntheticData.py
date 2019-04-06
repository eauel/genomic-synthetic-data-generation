import msprime
import allel
import numpy as np
import dask.array as da

ZARR_PATH = './output.zarr'


def generate_coalescent_synthetic_data(length=1e7, mu=3.5e-9, rrate=1e-8, sample_size=1000, Ne=1, ploidy=2, seed=57):
    """
        Function credits: Nick Harding
        Reference URL: https://hardingnj.github.io/2017/08/23/power-of-correct-tools.html
    """
    tree_sequence = msprime.simulate(length=length,
                                     recombination_rate=rrate,
                                     mutation_rate=mu,
                                     sample_size=sample_size,
                                     Ne=Ne,
                                     random_seed=seed)

    # Print the number of mutations in tree sequence
    print("Simulated ", tree_sequence.get_num_mutations(), "mutations")

    V = np.zeros(shape=(tree_sequence.get_num_mutations(), tree_sequence.get_sample_size()), dtype=np.int8)

    for variant in tree_sequence.variants():
        V[variant.index] = variant.genotypes

    # for variant in tree_sequence.variants():
    #    print(variant.index, variant.position, variant.genotypes, sep="\t")

    ht = allel.HaplotypeArray(V)
    gt = ht.to_genotypes(ploidy=ploidy)
    gt = allel.GenotypeDaskArray(gt)  # Convert to underlying Dask array

    print('Num Allele Calls: {}'.format(gt.n_allele_calls))
    print('Num Calls: {}'.format(gt.n_calls))
    print('Num Samples: {}'.format(gt.n_samples))
    print('Num Variants: {}'.format(gt.n_variants))

    # Save the genotype data to zarr store
    print('Saving data to zarr directory')
    gt = gt.rechunk((65536, 64, ploidy))
    da.to_zarr(arr=gt, url=ZARR_PATH, component='calldata/GT', overwrite=True)

    print('done')


if __name__ == '__main__':
    generate_coalescent_synthetic_data(length=22553061,  # Number of bases
                                       mu=3.5e-9,
                                       rrate=1e-8,
                                       sample_size=100000 * 2,
                                       Ne=1000,  # Effective diploid population size
                                       ploidy=2,
                                       seed=57)
