from tabnanny import check
import pandas as pd
import numpy as np
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
import json


def get_sample_def(actual_values, predicted_values, fraction):
    return pd.DataFrame({"actual": actual_values, "predicted": predicted_values}).sample(frac=fraction, replace=False)


def round_df(input_df, n_decimial):
    df = input_df.copy()
    return df.applymap(lambda x: round(x, n_decimial))


def get_shap_sample_values_arr(df_shap, y_scaled):
    n = df_shap.shape[0]
    ones=np.ones(n)
    # rand=np.random.rand(n)
    return [np.c_[df_shap[col].values, ones+i, y_scaled.reshape(-1,1)].tolist() for i, col in enumerate(df_shap.columns)]


def generate_ml_and_shap_data(df_js):
    df = pd.DataFrame(json.loads(df_js))

    output_column_name = 'output'
    X = df.loc[:, df.columns != output_column_name]
    y = df.iloc[:, df.columns == output_column_name]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    xgb_reg = XGBRegressor(n_estimators=20)
    xgb_reg.fit(X_train, y_train)
    y_train_pred = xgb_reg.predict(X_train)
    y_test_pred = xgb_reg.predict(X_test)
    y_all_pred=xgb_reg.predict(X)
    

    model_r2_train_data = round(r2_score(y_train, y_train_pred), 3)
    model_r2_test_data = round(r2_score(y_test, y_test_pred), 3)

    df_train_pred_sample = round_df(get_sample_def(
        y_train[output_column_name].values, y_train_pred, 0.2), 4)

    df_test_pred_sample = round_df(get_sample_def(
        y_test[output_column_name].values, y_test_pred, 0.3), 4)

    model_pred_train_data = df_train_pred_sample.values.tolist()
    model_pred_test_data = df_test_pred_sample.values.tolist()

    explainer = shap.TreeExplainer(xgb_reg)
    shap_values = explainer.shap_values(X_train)

    df_train_shapValues = pd.DataFrame(
        shap_values, columns=X_train.columns.tolist())

    df_train_shapValues_sample = round_df(
        df_train_shapValues.sample(frac=0.4, replace=False), 4)
    
    y_train_pred_scaled=y_train_pred/(max(y_train_pred)-min(y_train_pred))
    
    y_train_pred_scaled_smaple=y_train_pred_scaled[df_train_shapValues_sample.index.values]
    

    shap_features = [col for col in df_train_shapValues.columns]
    
    shap_values_test = explainer.shap_values(X_test)
    
    df_XGB = pd.DataFrame(y_test_pred, index=X_test.index, columns=['prediction'])

    df_SHAP = pd.DataFrame(shap_values_test,
                        index=X_test.index,
                        columns=[f'{feature} contrib.' for feature in X_test.columns])
    df_XGB_SHAP = pd.concat([df_XGB, df_SHAP], axis=1)
    
    df_merged =round_df(pd.merge(left=X_test,
                         right=df_XGB_SHAP,
                         how='left',
                         left_index=False,
                         right_index=True,
                         left_on=X_test.index,
                         right_on=df_XGB_SHAP.index),4)
    #df_merged=round_df(df_merged,4)
    
    df_report_json=df_merged.sample(frac=0.3).to_json(orient='split')
    print(df_report_json)
    print(shap_features)
    
    
    shap_values_sample_arr = get_shap_sample_values_arr(
        df_train_shapValues_sample,y_train_pred_scaled_smaple)
    feature_importance = np.abs(shap_values).mean(0).tolist()
    #print(shap_values_sample_arr)
    

    return {"model": {"r2": {"train_data": model_r2_train_data, "test_data": model_r2_test_data},
                      "prediction": {"train_data": model_pred_train_data, "test_data": model_pred_test_data}
                      },
            "shap": {"features": shap_features,
                     "sample_values": shap_values_sample_arr,
                     "feature_importance": feature_importance,
                     "alaki":df_report_json
                     }
            }
