import msprime
import allel
import numpy as np
import dask.array as da

ZARR_PATH = './output.zarr'


def generate_coalescent_synthetic_data(pop_cfg, dem_hist, num_bases=1e7, mu=3.5e-9, rrate=1e-8,
                                       ploidy=2, seed=57):
    """
        Function credits: Nick Harding
        Reference URL: https://hardingnj.github.io/2017/08/23/power-of-correct-tools.html
    """
    tree_sequence = msprime.simulate(population_configurations=pop_cfg,
                                     demographic_events=dem_hist,
                                     length=num_bases,
                                     recombination_rate=rrate,
                                     mutation_rate=mu,
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

    # Save the genotype data to zarr store
    print('Saving data to zarr directory')
    gt_output = da.from_array(gt, chunks=(65536, 64, ploidy))
    da.to_zarr(arr=gt_output, url=ZARR_PATH, component='calldata/GT', overwrite=True)

    print('done')


if __name__ == '__main__':
    pop_history = [
        msprime.PopulationParametersChange(time=20, growth_rate=-0.25, population_id=0),
        msprime.PopulationParametersChange(time=40, growth_rate=-0.004, population_id=0),
        msprime.PopulationParametersChange(time=1000, growth_rate=0, population_id=0)]

    pop_config = [msprime.PopulationConfiguration(
        sample_size=100000 * 2, initial_size=1000, growth_rate=0)]

    dp = msprime.DemographyDebugger(population_configurations=pop_config,
                                    demographic_events=pop_history)
    dp.print_history()

    generate_coalescent_synthetic_data(pop_config,
                                       pop_history,
                                       num_bases=50e6,
                                       mu=3.5e-9,
                                       rrate=1e-8,
                                       ploidy=2,
                                       seed=57)
