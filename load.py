from torch_geometric.data import Dataset, Data
import numpy as np
import torch


class LatticeDataset(Dataset):
    def __init__(self, input_lattice_bases, output_lattice_bases):
        super(LatticeDataset, self).__init__()
        self.data_list = [
            create_lattice_graph(torch.tensor(input_basis), torch.tensor(output_basis))
            for input_basis, output_basis in zip(
                input_lattice_bases, output_lattice_bases
            )
        ]

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


def create_lattice_graph(input_lattice_basis, output_lattice_basis):
    """
    Create a PyTorch Geometric Data object from input and output lattice basis matrices.

    Args:
        input_lattice_basis (torch.Tensor): A 2D tensor representing the input (unreduced) lattice basis matrix.
        output_lattice_basis (torch.Tensor): A 2D tensor representing the output (reduced) lattice basis matrix.

    Returns:
        torch.geometric.data.Data: A PyTorch Geometric Data object representing the lattice.
    """
    # Convert the input and output lattice basis matrices to floating-point tensors
    input_lattice_basis = input_lattice_basis.float()
    output_lattice_basis = output_lattice_basis.float()

    # Compute additional features for the input lattice
    input_gram_matrix = torch.matmul(input_lattice_basis.T, input_lattice_basis)
    input_determinant = torch.det(input_lattice_basis)
    input_condition_number = torch.linalg.cond(input_lattice_basis)

    # Compute additional features for the output lattice
    output_gram_matrix = torch.matmul(output_lattice_basis.T, output_lattice_basis)
    output_determinant = torch.det(output_lattice_basis)
    output_condition_number = torch.linalg.cond(output_lattice_basis)

    # Represent the lattices as a graph
    num_nodes = input_lattice_basis.size(0)
    node_features = torch.cat(
        [
            input_lattice_basis.flatten(),
            input_gram_matrix.flatten(),
            torch.tensor([input_condition_number, input_determinant]),
            output_lattice_basis.flatten(),
            output_gram_matrix.flatten(),
            torch.tensor([output_condition_number, output_determinant]),
        ],
        dim=0,
    )
    print(node_features.shape)
    edge_index = torch.tensor(
        [[i, j] for i in range(num_nodes) for j in range(num_nodes)], dtype=torch.long
    ).T

    # Create a PyTorch Geometric Data object
    data = Data(x=node_features, edge_index=edge_index)
    return data
