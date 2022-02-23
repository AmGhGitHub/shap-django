from celery import shared_task
from .util.generate_var_dist import VarDf, VarHistogram
from .util.ml_shap import generate_ml_and_shap_data


@shared_task
def gen_results(sample_size, lst_variables, repeated_rows_pct, latex_eq):
    var_df = VarDf(sample_size, lst_variables,
                   repeated_rows_pct, latex_eq)

    df_with_nulls, df_without_nulls = var_df.df_with_null_values, var_df.df_without_null_values

    var_hist = VarHistogram(df_with_nulls)
    
    df_without_nulls_js=df_without_nulls.to_json()

    data={"Histogram data": var_hist.hist_data,
          "ML-SHAP data":generate_ml_and_shap_data(df_without_nulls_js)}
    return data