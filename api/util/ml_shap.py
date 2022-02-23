from dataclasses import replace
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
    
    model_train_pred=df_train_pred_sample.values.tolist()
    model_test_pred=df_test_pred_sample.tolist()

    explainer = shap.TreeExplainer(xgb_reg)
    shap_values = explainer.shap_values(X_train)

    df_train_shapValues = pd.DataFrame(
        shap_values, columns=X_train.columns.tolist())
        
    df_train_shapValues_sample=df_train_shapValues.sample(frac=0.4,replace=False)
    
    
    
    
    
    vals= np.abs(shap_values).mean(0)
    feature_importance = pd.DataFrame(list(zip(X_train.columns,vals)),columns=['col_name','feature_importance_vals'])

    shap_features=[col for col in df_train_shapValues.columns]
    shap_values_sample=[df_train_shapValues_sample[col].values.tolist() for col in df_train_shapValues_sample.columns]
    shap_feature_importance=np.abs(shap_values).mean(0).tolist()
    # print(shap_features)
    # print(shap_values_sample)
    # print(shap_feature_importance)

    return {"model_r2": {"train_data": r2_model_train, "test_data": r2_model_test},
            "model_prediction":{"train_data": model_train_pred,"test_data": model_test_pred},
            "shap":{"features":shap_features, 
                    "sample_values":shap_values_sample,
                    "shap_feature_importance":shap_feature_importance,
                    "values":df_train_shapValues_sample.to_json(orient='values', double_precision=4),
                    "feature_importance": feature_importance.to_json(orient='split', double_precision=4)}
            }
