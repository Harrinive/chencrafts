import numpy as np
import qutip as qt
import scqubits as scq
from chencrafts.cqed.qt_helper import ket_in_basis

from typing import List, Tuple


class FloquetBasis(qt.FloquetBasis):

    @staticmethod
    def _closest_state( 
        states: List[qt.Qobj],
        target: qt.Qobj,
    ) -> Tuple[int, qt.Qobj]:
        """
        Find the state in `states` that has the largest overlap with `target`.

        Returns
        -------
        Tuple[int, qt.Qobj]
            The index of the closest state and the state found.
        """
        overlaps = np.array([np.abs(state.overlap(target)) ** 2 for state in states])
        return np.argmax(overlaps), states[np.argmax(overlaps)]

    def floquet_lookup(
        self,
        t: float = 0, 
        evecs: List[qt.Qobj] | None = None, 
        threshold = 0.5
    ) -> List[int | None]:
        """
        When a floquet state have overlap with a undriven state larger than `threshold`, 
        it's considered to be the dressed state corresponding to the undriven state.

        Parameters
        ----------
        t: float, optional
            Time at which the Floquet modes are evaluated. Default is 0.
        evecs : list[qt.Qobj], optional
            Undriven eigenvectors. If None, the function will generate a set of 
            Fock basis with the same dimension as the Floquet eigenvectors.
        threshold : float, optional
            Overlap threshold.

        Returns
        -------
        list[int | None]
            The index of the floquet state corresponding to the undriven state index.
        """
        fevecs = self.mode(t)

        dim = fevecs[0].shape[0]
        if evecs is None:
            evecs = [qt.basis(dim, i) for i in range(dim)]

        lookup = []
        for evec in evecs:
            overlaps = np.array([np.abs(evec.overlap(fevec)) ** 2 for fevec in fevecs])
            if np.any(overlaps > threshold):
                lookup.append(np.argmax(overlaps))
            else:
                lookup.append(None)

        return lookup

    def floquet_state_component(
        self,
        t: float,
        state_label: int,
        bare_evecs: List[qt.Qobj] | None = None,
        provide_bare_label: bool = False,
        return_amplitude: bool = False,
        truncate: int | None = None,
    ) -> Tuple[List[int], List[float]]:
        """
        For a dressed state with bare_label, will return the bare state conponents and the 
        corresponding occupation probability. 
        They are sorted by the probability in descending order.

        Parameters
        ----------
        t: float
            Time at which the Floquet modes are evaluated.
        state_label: int
            The state label of the Floquet state. It can also be a bare label if 
            provide_bare_label is set to True.
        bare_evecs: Tuple, optional
            A list of eigenstates of the bare system. If not provided,
            the function will generate a set of Fock basis.
        provide_bare_label: bool
            If True, the state_label will be treated as a bare label. Otherwise,
            it will be treated as a Floquet state label.
        return_amplitude: bool
            If True, the function will return the quantum probability amplitude
            of the state component. Otherwise, return a probability.
        truncate:
            The number of components to be returned. If None, all components 
            will be returned.

        Returns
        -------
        Tuple[List[int], List[float]]
            A tuple of two lists, the first list contains the bare state labels, 
            and the second list contains the corresponding occupation probabilities.
        """
        floquet_evecs = self.mode(t)
        dim = floquet_evecs[0].shape[0]

        if bare_evecs is None:
            bare_evecs = [qt.basis(dim, i) for i in range(dim)]

        if provide_bare_label:
            lookup = self.floquet_lookup(t, bare_evecs, threshold=scq.settings.OVERLAP_THRESHOLD)
            floquet_state_label = lookup[state_label]
            if floquet_state_label is None:
                raise ValueError("No corresponding Floquet state found for the given bare state.")
        else:
            floquet_state_label = state_label

        f_evec = floquet_evecs[floquet_state_label]
        f_evec = ket_in_basis(f_evec, bare_evecs)
        prob = np.abs(f_evec.full()[:, 0])**2
        largest_occupation_label = np.argsort(prob)[::-1]

        bare_label_list = []
        prob_list = []
        for idx in range(f_evec.shape[0]):
            bare_label = int(largest_occupation_label[idx])

            if return_amplitude:
                p = f_evec.full()[bare_label, 0]
            else:
                p = prob[bare_label]

            bare_label_list.append(bare_label)
            prob_list.append(p)

        if truncate is not None:
            bare_label_list = bare_label_list[:truncate]
            prob_list = prob_list[:truncate]

        return bare_label_list, prob_list
    
    def prop_floquet(self, t: float, t0: float = 0.0) -> qt.Qobj:
        """
        Slow propagator by Floquet Hamiltonian exp[-i H_F (t - t0)], 
        following the Viebahn 2020's notation.

        Parameters
        ----------
        t: float
            Final time for the evolution.
        t0: float, optional
            Initial time for the evolution, also serves as the evaluation 
            time for the Floquet modes. Default is 0.
        """
        fevals = self.e_quasi
        fevecs = self.mode(t0)
        prop_floquet = sum(
            [
                vec * np.exp(-1j * val * (t - t0)) * vec.dag()
                for vec, val in zip(fevecs, fevals)
            ]
        )
        return prop_floquet
    
    def ham_floquet(self) -> qt.Qobj:
        """
        Floquet Hamiltonian H_F (t0 = 0), following the Viebahn 2020's notation.

        Parameters
        ----------
        t: float
            Final time for the evolution.
        t0: float, optional
            Initial time for the evolution, also serves as the evaluation 
            time for the Floquet modes. Default is 0.
        """
        fevals = self.e_quasi
        fevecs = self.mode(0)
        prop_floquet = sum(
            [
                vec * val * vec.dag()
                for vec, val in zip(fevecs, fevals)
            ]
        )
        return prop_floquet
    
    def prop_kick(self, t, t0 = 0.0):
        """
        Stroboscopic kick propagator U_K = exp[-i K [t0] (t)], following the
        Viebahn 2020's notation.

        Parameters
        ----------
        t: float
            Final time for the evolution.
        t0: float, optional
            Initial time for the evolution, also serves as the evaluation 
            time for the Floquet modes. Default is 0.
        """
        T = self.T
        U_K_t = self.U(t % T, 0) 
        U_K_t0 = self.U(t0 % T, 0)

        return U_K_t * self.prop_floquet(t % T, t0 % T).dag() * U_K_t0.dag()
    
    def propagator(self, t, t0 = 0.0):
        """
        Full propagator for the time dynamics. It's a speed-up version of
        self.U(t, t0).

        Parameters
        ----------
        t: float
            Final time for the evolution.
        t0: float, optional
            Initial time for the evolution, also serves as the evaluation 
            time for the Floquet modes. Default is 0.
        """
        prop_floquet = self.prop_floquet(t, t0)
        prop_kick_ = self.prop_kick(t, t0)

        return prop_kick_ * prop_floquet

