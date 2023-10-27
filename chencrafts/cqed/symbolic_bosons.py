from math import factorial

from sympy import (Add, Mul, S, ff)
from sympy.physics.secondquant import CreateBoson

def normal(
    e,
    **kwargs
):
    """
    Returns the normal ordered equivalent of an expression using McCoy's formula.

    It is directly copied from
    https://github.com/xiaoxuisaac/driven-boson-recursion
    """

    if not e:
        return S.Zero

    opts = {
        'simplify_kronecker_deltas': False,
        'expand': True,
        'simplify_dummies': False,
        'keep_only_fully_contracted': False,
        'lam': 1
    }
    opts.update(kwargs)

    # break up any NO-objects, and evaluate commutators
    e = e.doit(wicks=True)

    # make sure we have only one term to consider
    e = e.expand()
    if isinstance(e, Add):
        return Add(*[ normal(term, **kwargs) for term in e.args])

    # For Mul-objects we can actually do something
    if isinstance(e, Mul):

        # we don't want to mess around with commuting part of Mul
        # so we factorize it out before starting recursion
        c_part = []
        string1 = []
        for factor in e.args:
            if factor.is_commutative:
                c_part.append(factor)
            else:
                string1.append(factor)
        n = len(string1)

        # catch trivial cases
        if n == 0:
            result = e
        elif n == 1:
            result = e
        else:  # non-trivial
            if n > 2:
                # recursive step
                if isinstance(string1[0], CreateBoson) \
                    or (string1[0].is_Pow and \
                        isinstance(string1[0].args[0], CreateBoson)):
                    # if the first term is ad^m
                    result =  string1[0] * normal(Mul(*string1[1:]), **kwargs)
                else:
                    # if the first term is a^m,
                    # bring the first two terms to normal order first
                    result =  normal(normal(Mul(*string1[:2]), **kwargs)*Mul(*string1[2:]), **kwargs)
            else: # n = 2,
                result = _normal(string1, opts['lam'])
            result = Mul(*c_part)*result

        if opts['expand']:
            result = result.expand()
        # if opts['simplify_kronecker_deltas']:
        #     result = evaluate_deltas(result)

        return result

    # there was nothing to do
    return e


def _normal(string1, lam = 1):
    # string1 conatains two items
    # this function only normal order expression with one kind of boson
    if isinstance(string1[0], CreateBoson) \
        or (string1[0].is_Pow and isinstance(string1[0].args[0], CreateBoson)):
            # trivial case, already normal ordered
            return string1[0]*string1[1]
    else:
        #nontrivial, mccoy
        result = S(0)

        if string1[0].is_Pow: a, a_power = string1[0].args
        else: a, a_power = string1[0], 1


        if string1[1].is_Pow: ad, ad_power = string1[1].args
        else: ad, ad_power = string1[1], 1

        if ad.args[0] != a.args[0]:
            raise Exception("expression to be normal ordered contains multiple \
                            kinds of bosons. Not Implemented")

        for k in range(min(ad_power, a_power)+1):
            result += ff(a_power, k)*ff(ad_power, k)/factorial(k)*\
                ad**(ad_power - k)*a**(a_power - k)*lam**k
    return result



