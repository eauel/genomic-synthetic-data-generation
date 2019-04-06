import msprime


def generate_coalescent_synthetic_data(pop_cfg, dem_hist, length=1e7, mu=3.5e-9, rrate=1e-8, seed=57):
    """
        Function credits: Nick Harding
        Reference URL: https://hardingnj.github.io/2017/08/23/power-of-correct-tools.html
    """
    tree_sequence = msprime.simulate(length=length,
                                     recombination_rate=rrate,
                                     mutation_rate=mu,
                                     random_seed=seed,
                                     population_configurations=pop_cfg,
                                     demographic_events=dem_hist)

    # Print the number of mutations in tree sequence
    # print("Simulated ", tree_sequence.get_num_mutations(), "mutations")


if __name__ == '__main__':
    dem_hist = [
        msprime.PopulationParametersChange(time=20, growth_rate=-0.25, population_id=0),
        msprime.PopulationParametersChange(time=40, growth_rate=0, population_id=0)
    ]

    pop_cfg = [
        msprime.PopulationConfiguration(sample_size=100, initial_size=1000, growth_rate=0)
    ]

    generate_coalescent_synthetic_data(pop_cfg, dem_hist)
