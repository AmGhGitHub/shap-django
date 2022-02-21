from celery import shared_task
from .util.generate_var_dist import VarDf, VarHistogram
from .util.ml_shap import gen_test_shap_plot


@shared_task
def gen_results(sample_size, lst_variables, repeated_rows_pct, latex_eq):
    var_df = VarDf(sample_size, lst_variables,
                   repeated_rows_pct, latex_eq)

    df_with_nulls, df_without_nulls = var_df.df_with_null_values, var_df.df_without_null_values

    var_hist = VarHistogram(df_with_nulls)
    
    df_without_nulls_js=df_without_nulls.to_json()
    # print(df_without_nulls_js)
    
    img=gen_test_shap_plot(df_without_nulls_js)
    

    hist_input_binSize_binCenters, hist_output_binSize_binCenters = var_hist.hist_input_data, var_hist.hist_output_data
    # print(hist_input_binSize_binCenters)
    return hist_input_binSize_binCenters, hist_output_binSize_binCenters#,img