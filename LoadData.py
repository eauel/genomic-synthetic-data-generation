import zarr
import allel


def open_zarr_dataset(zarr_path):
    store = zarr.DirectoryStore(zarr_path)
    callset = zarr.Group(store=store, read_only=True)
    return callset


def get_genotype_data(callset):
    genotype_ref_name = ''

    # Ensure 'calldata' is within the callset
    if 'calldata' in callset:
        # Try to find either GT or genotype in calldata
        if 'GT' in callset['calldata']:
            genotype_ref_name = 'GT'
        elif 'genotype' in callset['calldata']:
            genotype_ref_name = 'genotype'
        else:
            return None
    else:
        return None

    gtz = callset['calldata'][genotype_ref_name]

    return allel.GenotypeDaskArray(gtz)


if __name__ == '__main__':
    callset = open_zarr_dataset('./output.zarr')

    gt = get_genotype_data(callset)
    print(gt)
