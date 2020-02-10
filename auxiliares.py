import sympy as sp
import numpy as np
import math
from sympy.parsing.sympy_parser import parse_expr, standard_transformations
from scipy.odr import Model
from numpy import mean, vectorize
from numpy.linalg import norm
from scipy.stats import sem
from uncertainties import unumpy as unp
from scipy.optimize import curve_fit
from scipy.odr import Model, RealData, ODR

# Funções auxiliares

def utri(a):
    return a/(2*math.sqrt(6))

def uret(a):
    return a/(2*math.sqrt(3))

def combinada(*args):
    return norm(args)

def funfórmula(pares, f, subs=None):
    if subs is None:
        subs = {}
    subs.update(pares)
    return float(f.evalf(16, subs=subs))

def vetórmula(pares, f, subs=None):
    return vectorize(lambda p: funfórmula(p, f, subs))(pares)

def parse(s):
    return parse_expr(s, transformations=standard_transformations)

def sigdig(v: float, u:float):
    if u == 0:
        return f'({v} ± 0)'
    sdig = -math.floor(math.log10(abs(u)))
    if sdig > 0:
        return f'({round(v, sdig)} ± {round(u, sdig)})'
    else:
        return f'({int(round(v, sdig))} ± {int(round(u, sdig))})'

def incerteza(f: sp.Expr) -> sp.Expr:
    uf = 0
    udict = {}
    for s in f.free_symbols:
        udict[s] = sp.Symbol('u' + str(s))
        uf += udict[s] ** 2 * sp.simplify(f.diff(s) ** 2)
    uf = sp.sqrt(uf)
    return uf


def incertezarápido(f: sp.Expr) -> sp.Expr:
    uf = 0
    udict = {}
    for s in f.free_symbols:
        udict[s] = sp.Symbol('u' + str(s))
        uf += udict[s] ** 2 * (f.diff(s) ** 2)
    uf = sp.sqrt(uf)
    return uf

def float_input(prompt=""):
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Sinto muito, mas eu só gosto de floats...")

def int_input(prompt=""):
    while True:
        try:
            return int(input(prompt))
        except ValueError:
            print("Sinto muito, mas eu só gosto de ints...")

mlinear = Model(lambda beta, x: beta[0]*x + beta[1])
flinear = lambda x, a, b : a*x + b
ljac = lambda x, a, b : np.transpose([x, np.ones_like(x)])

def ajuste_odr(x, y, ux, uy):
    data = RealData(x, y, sx=ux, sy=uy)
    odr = ODR(data, mlinear, beta0=[-1000., 1.], ndigit=20)
    ajuste = odr.run()
    a, b = ajuste.beta
    ua, ub = np.sqrt(np.diag(ajuste.cov_beta))
    ajuste.pprint()
    return a, ua, b, ub

def ajuste_mmq(x, y, uy):
    p, c = curve_fit(flinear, x, y, sigma=uy, absolute_sigma=True, jac=ljac)
    a, b = p
    ua, ub = np.sqrt(np.diag(c))
    return a, ua, b, ub

def ajuste_odr_u(x, y):
    data = RealData(unp.nominal_values(x), unp.nominal_values(y), sx=unp.std_devs(x), sy=unp.std_devs(y))
    odr = ODR(data, mlinear, beta0=[1., 1.], ndigit=20)
    ajuste = odr.run()
    a, b = ajuste.beta
    ua, ub = np.sqrt(np.diag(ajuste.cov_beta))
    return a, ua, b, ub