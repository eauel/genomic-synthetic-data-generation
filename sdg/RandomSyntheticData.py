import numpy as np
import dask.array as da

DEFAULT_CHUNK_LENGTH = 2 ** 16
DEFAULT_CHUNK_WIDTH = 2 ** 6


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
                                      chunks=(DEFAULT_CHUNK_LENGTH, DEFAULT_CHUNK_WIDTH, ploidy))

    # Fill in random ones based on heterozygosity rate
    genetic_data = da.map_blocks(generate_random_genotype_data,
                                 genetic_data,
                                 dtype=np.int8,
                                 ones_rate=heterozygosity_rate)
    return genetic_data
