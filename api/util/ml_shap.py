import pandas as pd
import numpy as np
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
import json


def get_sample_def(actual_values, predicted_values, fraction):
    return pd.DataFrame({"actual": actual_values, "predicted": predicted_values}).sample(frac=fraction, replace=False)


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
    r2_model_train = r2_score(y_train, y_train_pred)
    r2_model_test = r2_score(y_test, y_test_pred)

    df_train_pred_sample = get_sample_def(
        y_train[output_column_name].values, y_train_pred, 0.2)
    df_test_pred_sample = get_sample_def(
        y_test[output_column_name].values, y_test_pred, 0.3)

    explainer = shap.TreeExplainer(xgb_reg)
    shap_values = explainer.shap_values(X_train)

    # df_train_shapValues = pd.DataFrame(
    #     shap_values, columns=X_train.columns.tolist())
    
    vals= np.abs(shap_values).mean(0)
    feature_importance = pd.DataFrame(list(zip(X_train.columns,vals)),columns=['col_name','feature_importance_vals'])
    # feature_importance.sort_values(by=['feature_importance_vals'],ascending=False,inplace=True)
    # print(feature_importance.head())

    return {"model_r2": {"train_data": r2_model_train, "test_data": r2_model_test},
            "model_prediction":{"train_data": df_train_pred_sample.to_json(orient='values'),"test_data": df_test_pred_sample.to_json(orient='values')},
            "shap":{"feature_importance": feature_importance.to_json(orient='split')}
            }
