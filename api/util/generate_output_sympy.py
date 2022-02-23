# based on
from latex2sympy2 import latex2sympy
from sympy import symbols, lambdify
import numpy
import os


def add_output_to_df(input_df, latex_expr,save_results=False):
    u, v, w, x, y, z = symbols('u v w x y z')
    expr = latex2sympy(latex_expr)

    df = input_df.copy()
    number_var = len(df.columns)

    output = None
    if number_var == 1:
        g = lambdify([u], expr, "numpy")
        output = g(df['u'].values)
    if number_var == 2:
        g = lambdify([u, v], expr, "numpy")
        output = g(df['u'].values, df['v'].values)
    if number_var == 3:
        g = lambdify([u, v, w], expr, "numpy")
        output = g(df['u'].values, df['v'].values, df['w'].values)
    if number_var == 4:
        g = lambdify([u, v, w, x], expr, "numpy")
        output = g(df['u'].values, df['v'].values,
                   df['w'].values, df['x'].values)
    if number_var == 5:
        g = lambdify([u, v, w, x, y], expr, "numpy")
        output = g(df['u'].values, df['v'].values,
                   df['w'].values, df['x'].values, df['y'].values)
    if number_var == 6:
        g = lambdify([u, v, w, x, y, z], expr, "numpy")
        output = g(df['u'].values, df['v'].values,
                   df['w'].values, df['x'].values, df['y'].values,  df['z'].values)

    df['output'] = output
    if save_results:
        df.to_excel('./api/util/data/df.xlsx')
    return df


if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    dict_value = {'u': np.random.randint(
        1, 10, 5), 'v': np.random.randint(-10, -1, 5), 'w': np.random.randint(50, 70, 5), 'x': np.random.randint(100, 200, 5)}
    df = pd.DataFrame(dict_value)
    # print(df)
    res = add_output_to_df(df, "u+v/w+1000/x")
    print(res)
