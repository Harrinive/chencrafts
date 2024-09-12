import numpy as np
import qutip as qt
from chencrafts.cqed.qt_helper import leakage_amount
from typing import List, Tuple

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False 


class SynthBase:
    def __init__(
        self, 
        original_U: np.ndarray | qt.Qobj, 
        target_U: np.ndarray | qt.Qobj,
    ):
        """
        Initialize the synthesis class.
        
        Parameters
        ----------
        original_U: np.ndarray | qt.Qobj
            The original 2-qubit gate.
        target_U: np.ndarray | qt.Qobj
            The target 2-qubit gate.
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is a optional dependency for block_diag module."
                "Please install it via 'pip install torch' or 'conda install "
                "pytorch torchvision -c pytorch'."
            )
        
        self.original_U = (
            original_U if isinstance(original_U, np.ndarray) 
            else original_U.full()
        )
        self.target_U = (
            target_U if isinstance(target_U, np.ndarray) 
            else target_U.full()
        )
        self.leakage = leakage_amount(qt.Qobj(self.original_U))

    @staticmethod
    def _stack_tensor(nd_list: List) -> "torch.Tensor":
        """
        Stack a nD list of 0D torch tensor into a nD torch tensor.
        
        Typical usage:
        >>> stack_tensor([
            [alpha, beta],
            [gamma, delta],
        ])
        >>> tensor([
            [alpha, beta],
            [gamma, delta],
        ])
        where alpha, beta, gamma, delta are 0D torch tensor.
        """
        # recursively stack the tensor
        if not isinstance(nd_list[0], list):
            # we are at the last dimension
            return torch.stack(nd_list)
        
        tensor = torch.stack(
            [
                SynthBase._stack_tensor(l) for l in nd_list
            ],
            dim=0
        )
        
        return tensor
    
    @staticmethod
    def _su2_matrix(theta: "torch.Tensor", phi: "torch.Tensor") -> "torch.Tensor":
        """
        Generate the SU(2) matrix for a single qubit.
        
        [
            [cos(theta/2), -exp(1j * phi) * sin(theta/2)],
            [exp(1j * phi).conj() * sin(theta/2), cos(theta/2)]
        ]
        
        Parameters
        ----------
        theta: torch.Tensor
            The theta parameter.
        phi: torch.Tensor
            The phi parameter.
            
        Returns
        -------
        torch.Tensor
            The SU(2) matrix.
        """
        cos_theta = torch.cos(theta / 2)
        sin_theta = torch.sin(theta / 2)
        exp_phi = torch.exp(1j * phi)
        return SynthBase._stack_tensor([
            [cos_theta, -exp_phi * sin_theta],
            [torch.conj(exp_phi) * sin_theta, cos_theta]
        ])

    @staticmethod
    def _kron_multi(*matrices: List["torch.Tensor"]) -> "torch.Tensor":
        """
        Compute the Kronecker product of multiple matrices.
        
        Parameters
        ----------
        *matrices: List[torch.Tensor]
            The matrices to be Kronecker producted.
            
        Returns
        -------
        torch.Tensor
            The Kronecker product of the matrices.
        """
        result = matrices[0]
        for matrix in matrices[1:]:
            result = torch.kron(result, matrix)
        return result

    @staticmethod
    def _su2_matrix_kron(
        theta: "torch.Tensor", 
        phi: "torch.Tensor", 
        num_qubits: int, 
        which_qubit: int,
    ) -> "torch.Tensor":
        """
        Generate the SU(2) matrix for a single qubit in the Kronecker product 
        basis of multi-qubit system.
        
        [
            [cos(theta/2), -exp(1j * phi) * sin(theta/2)],
            [exp(1j * phi).conj() * sin(theta/2), cos(theta/2)]
        ]
        
        Parameters
        ----------
        theta: torch.Tensor
            The theta parameter.
        phi: torch.Tensor
            The phi parameter.
        num_qubits: int
            The number of qubits.
        which_qubit: int
            The index of the qubit to apply the SU(2) matrix to.
            
        Returns
        -------
        torch.Tensor
            The SU(2) matrix for the qubit.
        """
        I = torch.eye(2, dtype=torch.complex128)
        I_list = [I] * num_qubits
        U = SynthBase._su2_matrix(theta, phi)
        I_list[which_qubit] = U
        return SynthBase._kron_multi(*I_list)
    
    @staticmethod
    def _process_fidelity(
        oprt: "torch.Tensor", 
        tgt: "torch.Tensor",
    ) -> "torch.Tensor":
        """
        Compute the process fidelity between two matrices.
        
        Parameters
        ----------
        oprt: torch.Tensor
            The operator matrix.
        tgt: torch.Tensor
            The target matrix.
            
        Returns
        -------
        torch.Tensor
            The process fidelity.
        """
        d = oprt.shape[0]
        return torch.abs(torch.trace(oprt @ tgt.T.conj())) ** 2 / d ** 2


class OneLayerSynth(SynthBase):
    params: np.ndarray
    leakage: float
    
    def __init__(
        self, 
        original_U: np.ndarray | qt.Qobj, 
        target_U: np.ndarray | qt.Qobj,
        params: np.ndarray | None = None,
        qubit_pair: Tuple[int, int] = (0, 1),
        num_qubits: int = 2,
    ):
        """
        Synthesize a new 2-qubit gate from a given 2-qubit gate with 4 
        single qubit gates.
        
        new_U = gate_A_2 @ gate_B_2 @ original_U @ gate_B_1 @ gate_A_1
        
        The single qubit gates are determined by optimizing the fidelity,
        which will be stored as params after calling the run() method.
        
        Parameters
        ----------
        original_U: jnp.ndarray
            The 2-qubit gate to be synthesized.
        target_U: np.ndarray | qt.Qobj
            The target 2-qubit gate.
        params: np.ndarray | None
            The length-8 array for gates A1, B1, A2, B2, with each 
            being a theta, phi pair. By default, it is zeros. After run()
            is called, it will be overwritten by the optimized parameters.
        qubit_pair: Tuple[int, int] = (0, 1)
            The qubit pair to be synthesized.
        num_qubits: int = 2
            The number of qubits (assume we are looking at a multi-qubit 
            system).
        """
        super().__init__(original_U, target_U)
        self.qA, self.qB = qubit_pair
        self.num_qubits = num_qubits
        if params is None:
            self.params = np.zeros(8, dtype=np.float64)
        else:
            self.params = params
        
    def _synth_1layer(
        self,
        params: "torch.Tensor", 
        original_U: "torch.Tensor",
    ):
        """
        Synthesize the 1-layer gate.
        
        Parameters
        ----------
        params: torch.Tensor
            The parameters for all the single qubit gates.
        original_U: torch.Tensor
            The original 2-qubit gate.
            
        Returns
        -------
        torch.Tensor
            The synthesized 2-qubit gate.
        """
        gate_A_1 = OneLayerSynth._su2_matrix_kron(params[0], params[1], self.num_qubits, self.qA)
        gate_B_1 = OneLayerSynth._su2_matrix_kron(params[2], params[3], self.num_qubits, self.qB)
        gate_A_2 = OneLayerSynth._su2_matrix_kron(params[4], params[5], self.num_qubits, self.qA)
        gate_B_2 = OneLayerSynth._su2_matrix_kron(params[6], params[7], self.num_qubits, self.qB)
        return gate_A_2 @ gate_B_2 @ original_U @ gate_B_1 @ gate_A_1

    def _cost_1layer(
        self,
        params: "torch.Tensor", 
        original_U: "torch.Tensor",
        target_U: "torch.Tensor",
    ):
        """
        Compute the cost function for the 1-layer gate, which is 1 - process_fidelity.
        
        Parameters
        ----------
        params: torch.Tensor
            The parameters for all the single qubit gates.
        original_U: torch.Tensor
            The original 2-qubit gate.
        target_U: torch.Tensor
            The target 2-qubit gate.
            
        Returns
        -------
        torch.Tensor
            The cost function = 1 - process_fidelity.
        """
        new_U = self._synth_1layer(params, original_U)
        return 1 - self._process_fidelity(new_U, target_U)
    
    def run(
        self,
        num_iter: int = 1000,
        tol: float = 1e-6,
        lr: float = 0.01,
    ):
        """
        Synthesize the 1-layer gate starting from zero initial parameters.
        
        Parameters
        ----------
        num_iter: int
            The number of iterations for the optimization.
        tol: float
            The tolerance for the optimization.
        lr: float
            The learning rate for the optimization.
            
        Returns
        -------
        np.ndarray
            The parameters for all the single qubit gates.
        """
        orig_U_tensor = torch.tensor(self.original_U, dtype=torch.complex128)
        target_U_tensor = torch.tensor(self.target_U, dtype=torch.complex128)
        
        init_params = torch.zeros(8, dtype=torch.float64, requires_grad=True)
        
        optimizer = torch.optim.Adam([init_params], lr=lr)
        
        # optimization loop
        for _ in range(num_iter):
            optimizer.zero_grad()
            loss = self._cost_1layer(init_params, orig_U_tensor, target_U_tensor)
            loss.backward()
            optimizer.step()
            
            if loss.item() < tol:
                break
            
        self.params = init_params.detach().numpy()
        
        return self.params
    
    @property
    def synth_U(self):
        """
        The synthesized 2-qubit gate.
        """
        result = self._synth_1layer(
            torch.tensor(self.params, dtype=torch.complex128), 
            torch.tensor(self.original_U, dtype=torch.complex128)
        )
        return result.detach().numpy()
    
    @property
    def original_fidelity(self):
        """
        The process fidelity between the original gate and the target gate.
        """
        return self._process_fidelity(
            torch.tensor(self.original_U, dtype=torch.complex128), 
            torch.tensor(self.target_U, dtype=torch.complex128)
        ).item()
    
    @property
    def synth_fidelity(self):
        """
        The process fidelity between the synthesized gate and the target gate.
        """
        return self._process_fidelity(
            torch.tensor(self.synth_U, dtype=torch.complex128), 
            torch.tensor(self.target_U, dtype=torch.complex128)
        ).item()