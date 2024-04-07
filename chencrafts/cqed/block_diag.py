import qutip as qt
import numpy as np
from scipy.linalg import expm, block_diag

from typing import List, Tuple

def _block_diag_ord1(
    H: qt.Qobj | np.ndarray | np.matrix, 
    dims: List[int] | np.ndarray[int],
    lr: float = 0.2,
    tol: float = 1e-10,
    num_iter: int = 10000,
) -> Tuple[qt.Qobj, qt.Qobj]:
    """
    1st order block diagonalization of a Hamiltonian. 

    Parameters
    ----------
    H : qt.Qobj | np.ndarray | np.matrix
        Hamiltonian to be block-diagonalized.
    dims : List[int] | np.ndarray[int]
        Dimensions of the blocks.
    previous_S : qt.Qobj | np.ndarray | np.matrix | None
        When running this algorithm iteratively, the previous generator 
        of the Schrieffer-Wolff transformation can be fed to this function
        so that the optimization can start from the previous result.
    lr : float
        Learning rate for the optimization.
    tol : float
        Acceptable tolerance for the optimization.
    num_iter : int
        Number of iterations for the optimization.

    Returns
    -------
    L, U : Tuple[qt.Qobj, qt.Qobj]
        L: block-diagonalized Hamiltonian.
        U: unitary transformation.
        Here L = U^dag H U and U^dag U = I.
    """
    try:
        import torch
    except ImportError:
        raise ImportError(
            "PyTorch is a optional dependency for block_diag module."
            "Please install it via 'pip install torch' or 'conda install pytorch'."
        )
    

    # type checking
    if isinstance(H, qt.Qobj):
        qobj_dims = H.dims    # for returning the result as a qobj
        H: np.ndarray = H.data.toarray()
    else:
        qobj_dims = None    # meaning that the user do not feed a qobj
    assert np.sum(dims) == H.shape[0]
    
    # break the Hamiltonian into blocks
    blocks = np.ndarray((len(dims), len(dims)), dtype=np.ndarray)
    cum_dims = np.concatenate([[0], np.cumsum(dims)])  # cumulative dimensions
    for i in range(len(dims)):
        for j in range(len(dims)):
            blocks[i, j] = H[
                cum_dims[i]:cum_dims[i+1], 
                cum_dims[j]:cum_dims[j+1]
            ]

    # unperturbed Hamiltonian & perturbation
    H0_np = np.matrix(block_diag(*blocks.diagonal()))
    H1_np = np.matrix(H - H0_np)

    # use pytorch to set up the block-diagonalization problem
    H0 = torch.tensor(H0_np, dtype=torch.complex128, requires_grad=False)
    H1 = torch.tensor(H1_np, dtype=torch.complex128, requires_grad=False)
    generator = torch.zeros_like(H0, dtype=torch.complex128, requires_grad=True)

    def objective(
        generator: torch.Tensor, 
    ) -> torch.Tensor:
        # off-diagonal element H1 - [generator, H0] --> 0
        off_diag_elem = H1 - (generator @ H0 - H0 @ generator)
        cost1 = torch.sum(torch.stack(
            [
                torch.norm(off_diag_elem[
                    cum_dims[i]:cum_dims[i+1], 
                    cum_dims[j]:cum_dims[j+1]
                ])**2 
                for i in range(len(dims)) for j in range(len(dims)) if i != j
            ]
        ))

        # # keep the generator anti-Hermitian
        # cost2 = torch.norm(generator + generator.t().conj())**2
            
        # keep the diag subspace untouched
        cost3 = torch.sum(torch.stack([
            torch.norm(generator[
                cum_dims[i]:cum_dims[i+1], 
                cum_dims[i]:cum_dims[i+1]
            ])**2 for i in range(len(dims))
        ]))

        return cost1 + cost3
    
    # Adam iteration
    optimizer = torch.optim.Adam(
        [generator], 
        lr=lr,
    )
    # optimize & solve the problem
    for i in range(num_iter):
        optimizer.zero_grad()  # Reset the gradients of the optimizer.
        loss = objective(generator)  # Calculate the loss function.
        loss.backward()  # Compute the gradients of the loss function with respect to the generator.
        optimizer.step()    # Perform a single optimization step.

        # # constrain the generator to be anti-Hermitian and zero-diagonal
        # generator_clone = generator.clone()
        # generator_clone = (generator_clone - generator_clone.t().conj())/2
        # for j in range(len(dims)):
        #     generator_clone[
        #         cum_dims[j]:cum_dims[j+1], 
        #         cum_dims[j]:cum_dims[j+1]
        #     ] = 0
        # generator.data = generator_clone.data

        if loss < tol:
            print(f'Converged at iteration {i}.')
            break

    unitary = expm(generator.detach().numpy())
    block_diag_H = unitary.T.conj() @ H @ unitary

    if qobj_dims is not None:
        block_diag_H = qt.Qobj(block_diag_H, dims=qobj_dims)
        unitary = qt.Qobj(unitary, dims=qobj_dims)

    return block_diag_H, unitary

def block_diagonalize(
    H: qt.Qobj | np.ndarray | np.matrix, 
    dims: List[int] | np.ndarray[int],
    order: int = 1,
    lr: float = 0.1,
    tol: float = 1e-6,
    num_iter: int = 10000,
) -> Tuple[qt.Qobj, qt.Qobj]:
    """
    Block diagonalization of a Hamiltonian up to arbitrary order of the 
    Schrieffer-Wolff transformation.

    Parameters
    ----------
    H : qt.Qobj | np.ndarray | np.matrix
        Hamiltonian to be block-diagonalized.
    dims : List[int] | np.ndarray[int]
        Dimensions of the blocks.
    order : int
        Order of the Schrieffer-Wolff transformation.
    lr : float
        Learning rate for the optimization.
    num_iter : int
        Number of iterations for the optimization.

    Returns
    -------
    L, U : Tuple[qt.Qobj, qt.Qobj]
        L: block-diagonalized Hamiltonian.
        U: unitary transformation.
        Here L = U^dag H U and U^dag U = I.
    """
    # iterate the 1st order block diagonalization
    H_ = qt.Qobj(H) if not isinstance(H, qt.Qobj) else H
    U_ = qt.Qobj(np.eye(H.shape[0]), dims=H_.dims)

    for i in range(order):
        H_, U_step = _block_diag_ord1(
            H_, 
            dims, 
            lr=lr, 
            tol=tol,
            num_iter=num_iter,
        )
        U_ = U_ * U_step

    return H_, U_