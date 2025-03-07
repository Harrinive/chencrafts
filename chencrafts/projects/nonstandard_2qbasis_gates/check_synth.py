__all__ = [
    'check_synth_weyl', 
    'check_synth_CNOT',
    'check_synth_SWAP',
    'synth_complement',
    'synth_SWAP_in_3',
    'in_not_synth_swapin3_region',
    'in_not_synth_czin2_region',
]

# code copied from https://github.com/SophLin/nonstandard_2qbasis_gates

import numpy as np
from .conversions import (
    weyl_to_logspec, rotate
)

logspec_CZ = np.array([1/4,1/4,-1/4,-1/4])
logspec_SWAP = np.array([1/4,1/4,1/4,-3/4])
logspec_SWAP_r = np.array([3/4, -1/4, -1/4, -1/4])
can_sqrtSWAP = np.array([1/4,1/4,1/4])
can_sqrtSWAPd = np.array([3/4,1/4,1/4])
can_B = np.array([1/2,1/4,0])
can_SWAP = np.array([1/2,1/2,1/2])
can_CZ = np.array([1/2,0,0])

# [r,k,a,b,c,d, N] in quantum Littlewood-Richardson coefficients such that N_{ab}^{c,d}(r,k)=1 and r+k = 4
qlr =   [[1, 3, [0], [0], [0], 0, 1],
         [1, 3, [0], [1], [1], 0, 1],
         [1, 3, [0], [2], [2], 0, 1],
         [1, 3, [0], [3], [3], 0, 1],
         [1, 3, [1], [1], [2], 0, 1],
         [1, 3, [1], [2], [3], 0, 1],
         [1, 3, [1], [3], [0], 1, 1],
         [1, 3, [2], [2], [0], 1, 1],
         [1, 3, [2], [3], [1], 1, 1],
         [1, 3, [3], [3], [2], 1, 1],
         [2, 2, [0, 0], [0, 0], [0, 0], 0, 1],
         [2, 2, [0, 0], [1, 0], [1, 0], 0, 1],
         [2, 2, [0, 0], [1, 1], [1, 1], 0, 1],
         [2, 2, [0, 0], [2, 0], [2, 0], 0, 1],
         [2, 2, [0, 0], [2, 1], [2, 1], 0, 1],
         [2, 2, [0, 0], [2, 2], [2, 2], 0, 1],
         [2, 2, [1, 0], [1, 0], [2, 0], 0, 1],
         [2, 2, [1, 0], [1, 0], [1, 1], 0, 1],
         [2, 2, [1, 0], [1, 1], [2, 1], 0, 1],
         [2, 2, [1, 0], [2, 0], [2, 1], 0, 1],
         [2, 2, [1, 0], [2, 1], [2, 2], 0, 1],
         [2, 2, [1, 0], [2, 1], [0, 0], 1, 1],
         [2, 2, [1, 0], [2, 2], [1, 0], 1, 1],
         [2, 2, [1, 1], [1, 1], [2, 2], 0, 1],
         [2, 2, [1, 1], [2, 0], [0, 0], 1, 1],
         [2, 2, [1, 1], [2, 1], [1, 0], 1, 1],
         [2, 2, [1, 1], [2, 2], [2, 0], 1, 1],
         [2, 2, [2, 0], [2, 0], [2, 2], 0, 1],
         [2, 2, [2, 0], [2, 1], [1, 0], 1, 1],
         [2, 2, [2, 0], [2, 2], [1, 1], 1, 1],
         [2, 2, [2, 1], [2, 1], [2, 0], 1, 1],
         [2, 2, [2, 1], [2, 1], [1, 1], 1, 1],
         [2, 2, [2, 1], [2, 2], [2, 1], 1, 1],
         [2, 2, [2, 2], [2, 2], [0, 0], 2, 1],
         [3, 1, [0, 0, 0], [0, 0, 0], [0, 0, 0], 0, 1],
         [3, 1, [0, 0, 0], [1, 0, 0], [1, 0, 0], 0, 1],
         [3, 1, [0, 0, 0], [1, 1, 0], [1, 1, 0], 0, 1],
         [3, 1, [0, 0, 0], [1, 1, 1], [1, 1, 1], 0, 1],
         [3, 1, [1, 0, 0], [1, 0, 0], [1, 1, 0], 0, 1],
         [3, 1, [1, 0, 0], [1, 1, 0], [1, 1, 1], 0, 1],
         [3, 1, [1, 0, 0], [1, 1, 1], [0, 0, 0], 1, 1],
         [3, 1, [1, 1, 0], [1, 1, 0], [0, 0, 0], 1, 1],
         [3, 1, [1, 1, 0], [1, 1, 1], [1, 0, 0], 1, 1],
         [3, 1, [1, 1, 1], [1, 1, 1], [1, 1, 0], 1, 1],#below this line is the added part
        [1, 3, [1], [0], [1], 0, 1],
        [1, 3, [2], [0], [2], 0, 1],
        [1, 3, [3], [0], [3], 0, 1],
        [1, 3, [2], [1], [3], 0, 1],
        [1, 3, [3], [1], [0], 1, 1],
        [1, 3, [3], [2], [1], 1, 1],
        [2, 2, [1, 0], [0, 0], [1, 0], 0, 1],
        [2, 2, [1, 1], [0, 0], [1, 1], 0, 1],
        [2, 2, [2, 0], [0, 0], [2, 0], 0, 1],
        [2, 2, [2, 1], [0, 0], [2, 1], 0, 1],
        [2, 2, [2, 2], [0, 0], [2, 2], 0, 1],
        [2, 2, [1, 1], [1, 0], [2, 1], 0, 1],
        [2, 2, [2, 0], [1, 0], [2, 1], 0, 1],
        [2, 2, [2, 1], [1, 0], [2, 2], 0, 1],
        [2, 2, [2, 1], [1, 0], [0, 0], 1, 1],
        [2, 2, [2, 2], [1, 0], [1, 0], 1, 1],
        [2, 2, [2, 0], [1, 1], [0, 0], 1, 1],
        [2, 2, [2, 1], [1, 1], [1, 0], 1, 1],
        [2, 2, [2, 2], [1, 1], [2, 0], 1, 1],
        [2, 2, [2, 1], [2, 0], [1, 0], 1, 1],
        [2, 2, [2, 2], [2, 0], [1, 1], 1, 1],
        [2, 2, [2, 2], [2, 1], [2, 1], 1, 1],
        [3, 1, [1, 0, 0], [0, 0, 0], [1, 0, 0], 0, 1],
        [3, 1, [1, 1, 0], [0, 0, 0], [1, 1, 0], 0, 1],
        [3, 1, [1, 1, 1], [0, 0, 0], [1, 1, 1], 0, 1],
        [3, 1, [1, 1, 0], [1, 0, 0], [1, 1, 1], 0, 1],
        [3, 1, [1, 1, 1], [1, 0, 0], [0, 0, 0], 1, 1],
        [3, 1, [1, 1, 1], [1, 1, 0], [1, 0, 0], 1, 1]]

def check_synth_logspec(
    logspec1: np.ndarray,
    logspec2: np.ndarray,
    logspec3: np.ndarray,
    tol: float = 0.,
    verbose: bool = False,
):
    check = True
    for r,k,a,b,c,d,_ in qlr:
        acc = d
        for i in range(r):
            acc -= logspec1[k+i-a[i]]
            acc -= logspec2[k+i-b[i]]
            acc += logspec3[k+i-c[i]]
        if acc < 0-tol: # the inequality is not satisfied
            if verbose:
                print("Fails with r,k,a,b,c,d = ",r,k,a,b,c,d)
                print(acc,"<0")
                check = False
            else:
                return False
    return check

def check_synth_weyl(
    w1: np.ndarray,
    w2: np.ndarray,
    w3: np.ndarray,
    tol: float = 0.,
):
    """
    Check whether gate A can be decomposed into 2 layers by gate B and gate C.
    
    Parameters
    ----------
    w1, w2, w3: np.ndarray
        Cartan coordinates of the three gates in the Weyl chamber.
    tol: float
        Tolerance for the synthesis condition.
        
    Returns
    -------
    bool
        True if the synthesis is possible, False otherwise.
    """
    ls1,_ = weyl_to_logspec(w1)
    ls2,_ = weyl_to_logspec(w2)
    ls3,_ = weyl_to_logspec(w3)
    ls1_r = rotate(ls1)
    ls2_r = rotate(ls2)
    ls3_r = rotate(ls3)
    eq1 = np.array_equal(ls1,ls1_r)
    eq2 = np.array_equal(ls2,ls2_r)
    eq3 = np.array_equal(ls3,ls3_r)
    flag = check_synth_logspec(ls1,ls2,ls3,tol=tol)
    if flag:
        return True
    if not eq1:
        flag = check_synth_logspec(ls1_r,ls2,ls3,tol=tol)
        if flag:
            return True
    if not eq2:
        flag = check_synth_logspec(ls1,ls2_r,ls3,tol=tol)
        if flag:
            return True
    if not eq3:
        flag = check_synth_logspec(ls1,ls2,ls3_r,tol=tol)
        if flag:
            return True
    if not eq1 and not eq2:
        flag = check_synth_logspec(ls1_r,ls2_r,ls3,tol=tol)
        if flag:
            return True
    if not eq1 and not eq3:
        flag = check_synth_logspec(ls1_r,ls2,ls3_r,tol=tol)
        if flag:
            return True
    if not eq2 and not eq3:
        flag = check_synth_logspec(ls1,ls2_r,ls3_r,tol=tol)
        if flag:
            return True
    if not eq1 and not eq2 and not eq3:
        return check_synth_logspec(ls1_r,ls2_r,ls3_r,tol=tol)
    return False

def check_synth_SWAP(
    w1: np.ndarray,
    w2: np.ndarray,
    tol: float = 0.,
):
    """
    Check if the SWAP gate can be decomposed into two two-qubit gates.
    
    Parameters
    ----------
    w1, w2: np.ndarray
        Cartan coordinates of the two gates in the Weyl chamber.
    tol: float
        Tolerance for the synthesis condition.
        
    Returns
    -------
    bool
        True if the SWAP gate can be decomposed into two two-qubit gates, False otherwise.
    """
    return check_synth_weyl(w1,w2,can_SWAP,tol=tol)

def check_synth_CNOT(
    w1: np.ndarray,
    w2: np.ndarray,
    tol: float = 0.,
):
    """
    Check if the CNOT gate can be decomposed into two two-qubit gates.
    
    Parameters
    ----------
    w1, w2: np.ndarray
        Cartan coordinates of the two gates in the Weyl chamber.
    tol: float
        Tolerance for the synthesis condition.
        
    Returns
    -------
    bool
        True if the CNOT gate can be decomposed into two two-qubit gates, False otherwise.
    """
    return check_synth_weyl(w1,w2,can_CZ,tol=tol)

def synth_complement(weyl):
    """
    Given a gate, output another gate s.t. they can get SWAP in 2 steps
    
    Parameters
    ----------
    weyl: np.ndarray
        Cartan coordinates of the gate in the Weyl chamber.
    
    Returns
    -------
    np.ndarray
        Cartan coordinates of the complement gate in the Weyl chamber.
    """
    if weyl[0] <= 0.5:
        v1 = can_sqrtSWAP - can_B
    else:
        v1 = can_sqrtSWAPd - can_B
    v1 = v1/np.linalg.norm(v1)
    v2 = weyl - can_B
    proj = can_B + v1*np.dot(v1,v2)
    return proj - (weyl - proj)

def synth_SWAP_in_3(weyl: np.ndarray, tol: float = 0.):
    """
    Check if the input gate can be decomposed into two layers of another gate.
    
    Parameters
    ----------
    weyl: np.ndarray
        Cartan coordinates of the gate in the Weyl chamber.
    
    Returns
    -------
    bool
        True if the gate can be decomposed into two layers of another gate, False otherwise.
    """
    comp = synth_complement(weyl) # 2 layers of the gate has to get us its complement
    return check_synth_weyl(weyl,weyl,comp,tol=tol)

def in_not_synth_swapin3_region(w: np.ndarray):
    """
    Check if the input point is in one of the 2 regions.
    The volume for {not SWAP in 3} can be divided into 4 tetrahedra.
    Here we check for the 2 that include I.
    
    Parameters
    ----------
    w: np.ndarray
        Cartan coordinates of the gate in the Weyl chamber.
    
    Returns
    -------
    bool
        True if the gate is in the region, False otherwise.
    """
    #p0 = np.array([0,0,0]) # I0
    p1 = np.array([1/2,0,0])#CNOT/CZ
    #p2 = np.array([1,0,0]) #I1
    p3 = np.array([1/4,1/4,0]) #sqrt(iSWAP) closer to I0
    p4 = np.array([3/4,1/4,0]) #sqrt(iSWAP) closer to I1
    p5 = np.array([1/6,1/6,1/6]) # other vertex, closer to I0
    p6 = np.array([5/6,1/6,1/6]) # other vertex, closer to I1
    normal1 = np.cross(p5 - p1, p3 - p1)
    normal2 = np.cross(p4 - p1, p6 - p1)
    return np.dot(w-p1,normal1) > 0 or np.dot(w-p1,normal2) > 0

def in_not_synth_czin2_region(w: np.ndarray):
    """
    Check if the input point is in one of the 2 regions.
    The volume for {not CZ/CNOT in 2} can be divided into 3 tetrahedra.
    Here we check for the 2 that include I.
    
    Parameters
    ----------
    w: np.ndarray
        Cartan coordinates of the gate in the Weyl chamber.
    
    Returns
    -------
    bool
        True if the gate is in the region, False otherwise.
    """
    p1 = np.array([1/4,0,0])
    p2 = np.array([3/4,0,0])
    p3 = np.array([1/4,1/4,0]) #sqrt(iSWAP) closer to I0
    p4 = np.array([3/4,1/4,0]) #sqrt(iSWAP) closer to I1
    p5 = np.array([1/4,1/4,1/4]) # sqrt(SWAP)
    p6 = np.array([3/4,1/4,1/4]) # sqrt(SWAP)^dag
    normal1 = np.cross(p5 - p1, p3 - p1)
    normal2 = np.cross(p4 - p2, p6 - p2)
    return np.dot(w-p1,normal1) > 0 or np.dot(w-p2,normal2) > 0
