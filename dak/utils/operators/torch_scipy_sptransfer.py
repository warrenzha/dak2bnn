import torch
from scipy.sparse import coo_array


def scipy_coo_to_torch_coo(scipy_coo_array):
    """
    convert scipy.sparse.coo_array to torch.sparse_coo_tensor
    """
    row = torch.tensor(scipy_coo_array.row)
    col = torch.tensor(scipy_coo_array.col)
    vals = torch.tensor(scipy_coo_array.data, dtype=torch.float32)
    torch_coo_tensor = torch.sparse_coo_tensor(indices=torch.vstack((row, col)), values=vals,
                                               size=scipy_coo_array.shape)
    return torch_coo_tensor


def torch_coo_to_scipy_coo(torch_coo_tensor):
    """
    convert torch.sparse_coo_tensor to scipy.sparse.coo_array
    """
    ids = torch_coo_tensor._indices().numpy()
    data = torch_coo_tensor._values().numpy()
    scipy_coo_array = coo_array((data, (ids[0, :], ids[1, :])), shape=list(torch_coo_tensor.shape))
    return scipy_coo_array
