import torch


def cc_design(deg, input_bd=[-1,1], dyadic_sort=True):
    """
    :param deg: degree of clenshaw curtis (# of points = 2^(deg) - 1)
    :param input_bd: [x_leftbd, x_rightbd], [2] size list
    :param dyadic_sort: if sort=True, return sorted incremental tensor, default=True

    "return: res: [-cos( 0*pi/ n ), -cos( 1*pi/ n ), ..., -cos( n*pi/ n ) ], where n = 2^deg
         
    """
    x_1 = input_bd[0]
    x_n = input_bd[1]
    n = 2**(deg)

    if dyadic_sort is True:
        res_basis = torch.empty(0)
        for i in range(1, deg+1):
            m_i = 2**i
            increment_set = - torch.cos( torch.pi * torch.arange(start=1, end=m_i, step=2) / m_i)
            res_basis = torch.cat((res_basis,increment_set), dim=0)
    else:
        res_basis = - torch.cos( torch.pi * torch.arange(1, n) / n)  # interval on [-1,1]
    
    res = res_basis*(x_n-x_1)/2 + (x_n+x_1)/2  # [n-1] siz tensor
    
    return res