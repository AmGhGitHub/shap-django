import json
import numpy as np
import pandas as pd
from .generate_output_sympy import add_output_to_df


class VarHistogram:
    def __init__(self, df) -> None:
        self.df = df.copy()
        self.sample_size = self.df.shape[0]
        self.hist_data = self.__generate_histogram_data()

    def __get_nbins(self):
        size = self.sample_size
        if size > 0.9e7:
            return 500
        if size > 0.9e6:
            return 400
        if size > 0.9e5:
            return 300
        if size > 0.9e4:
            return 200
        if size > 0.9e3:
            return 50
        if size > 0.9e2:
            return 25
        return 10

    def __calc_bins_centers(self, bins):
        bin_w = (max(bins) - min(bins)) / (len(bins) - 1)
        return np.arange(min(bins) + bin_w / 2.0, max(bins), bin_w)

    def __generate_histogram_data(self):
        df = self.df.copy()
        hist_input_data = []
        hist_output_data = []

        for col in df.columns[:-1]:
            vals = df[col]
            vals = vals[~np.isnan(vals)]  # remove null values
            hist, bins = np.histogram(vals, self.__get_nbins())
            bins_centers = self.__calc_bins_centers(bins)
            hist_input_data.append(
                {"bin_size": hist.tolist(), "bin_centers": list(
                    np.round(bins_centers, 3))}
            )

        vals = df.iloc[:, -1:]  # get the output column
        vals = vals[~np.isnan(vals)]  # remove null values
        hist, bins = np.histogram(vals, self.__get_nbins())
        bins_centers = self.__calc_bins_centers(bins)
        hist_output_data.append(
            {"bin_size": hist.tolist(), "bin_centers": list(
                np.round(bins_centers, 3))}
        )
        hist_data={'inputs':hist_input_data,'output':hist_output_data}
        # print("hist_data:",hist_data)
        return hist_data


class VarDf:
    def __init__(self, sample_size, lst_var_specs, repeated_row_pct, latex_eq) -> None:
        self.sample_size = sample_size
        self.lst_var_specs = lst_var_specs
        self.lst_null_pct = [var["null_pct"] for var in self.lst_var_specs]
        self.repeated_row_frc = float(repeated_row_pct)/100
        self.latex_eq = latex_eq
        self.df_without_null_values, self.df_with_null_values = self.__generate_df_vars()

    def __get_df_with_nulls(self, input_df):
        df = input_df.copy()
        nrows = df.shape[0]
        max_ind = nrows - 1
        null_size = int(self.repeated_row_frc * nrows)
        null_inds = self.__get_null_indices(max_ind, null_size)
        df.loc[null_inds] = np.nan
        df = df.fillna(method="ffill")

        for i, col in enumerate(df.columns[:-1]):  # exclude output columns
            null_size = int((float(self.lst_null_pct[i]) / 100.0) * nrows)
            null_inds = self.__get_null_indices(max_ind, null_size)
            df.loc[null_inds, col] = np.nan

        return df

    def __generate_df_vars(self):
        dict_data = {}
        for var in self.lst_var_specs:
            column_name = var["letter"]
            distributio = var["distribution"]
            p0 = float(var["param0"])
            p1 = float(var["param1"])
            p2 = float(var["param2"])

            if distributio == "normal":
                mu = float(p0)
                sigma = float(p1)
                x_vals = np.random.normal(mu, sigma, self.sample_size)
            if distributio == "uniform":
                left = float(p0)
                right = float(p1)
                x_vals = np.random.uniform(left, right, self.sample_size)
            if distributio == "triangular":
                left = float(p0)
                mode = float(p1)
                right = float(p2)
                x_vals = np.random.triangular(
                    left, mode, right, self.sample_size)

            dict_data[column_name] = x_vals

        df_org = pd.DataFrame(dict_data)
        df_org_with_output = add_output_to_df(df_org, self.latex_eq)
        # print(df_org_with_output)
        # df_with_null_values = self.__get_df_with_nulls(df_org)
        df_with_output_with_null_values = self.__get_df_with_nulls(
            df_org_with_output)
        # w for with and wo for without
        return df_org_with_output, df_with_output_with_null_values

    def __get_null_indices(self, max_index, number_of_null_values):
        return np.random.choice(np.arange(1, max_index), size=number_of_null_values, replace=False)


if __name__ == "__main__":
    vars = [
        {
            "id": 0,
            "letter": "u",
            "distribution": "normal",
            "null_pct": 10,
            "param0": 0,
            "param1": 1,
            "param2": 0,
        },
        {
            "id": 1,
            "letter": "v",
            "distribution": "uniform",
            "null_pct": 7,
            "param0": -1,
            "param1": 1.5,
            "param2": 0,
        },
        {
            "id": 3,
            "letter": "w",
            "distribution": "normal",
            "null_pct": 12,
            "param0": 0.2,
            "param1": 0.5,
            "param2": 0,
        },
        {
            "id": 4,
            "letter": "x",
            "distribution": "triangular",
            "null_pct": 15,
            "param0": -5,
            "param1": 0,
            "param2": 5,
        },
    ]

    sample_size = 200
    latex_eq = "u+v+w+x"

    var_df = VarDf(sample_size, vars, 20, latex_eq)
    df_without_nulls, df_with_nulls = var_df.df_without_null_values, var_df.df_with_null_values
    # print("Without:\n",df_without_nulls)
    # print("With:\n",df_with_nulls)
    hist = VarHistogram(df_with_nulls)
    print(hist.hist_input_data)
    print(hist.hist_output_data)
    # df_without_nulls = var_df.df_without_null_values
    # hist = VarHistogram(df_without_nulls)
    # # print(hist.hist_data)

    # print(df_with_nulls)
    # hist = VarHistogram(df_with_nulls)
    # print(hist.hist_input_data)
    # print(hist.hist_output_data)
    # df_without_nulls = var_df.df_without_null_values
    # hist = VarHistogram(df_without_nulls)
    # # print(hist.hist_data)
