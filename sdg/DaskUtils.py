from distributed import Client

def connect_dask(address, port):
    # Connect to Dask scheduler
    print('[Dask Utils] Connecting to Dask scheduler.')
    client = Client('{}:{}'.format(address, port))
    return client