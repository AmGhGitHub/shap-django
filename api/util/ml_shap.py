import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
import json


def generate_ml_and_shap_data(df_js):
    df = pd.DataFrame(json.loads(df_js))
    
    output_column_name = 'output'
    X = df.loc[:, df.columns != output_column_name]
    y = df.iloc[:, df.columns == output_column_name]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    xgb_reg = XGBRegressor(n_estimators=20)
    xgb_reg.fit(X_train, y_train)
    y_train_prd = xgb_reg.predict(X_train)
    y_test_prd = xgb_reg.predict(X_test)
    r2_model_train = r2_score(y_train, y_train_prd)
    r2_model_test = r2_score(y_test, y_test_prd)

    df_prd_training = pd.DataFrame({"actual": y_train[output_column_name].values,
                                    "predicted": y_train_prd,
                                    "set": ["train"]*len(y_train)})

    df_prd_test = pd.DataFrame(
        {"actual": y_test[output_column_name].values, "predicted": y_test_prd, "set": ["test"]*len(y_test)})

    df_prediction = pd.concat(
        [df_prd_training, df_prd_test], ignore_index=True)

    explainer = shap.TreeExplainer(xgb_reg)
    shap_values = explainer.shap_values(X_train)

    df_shap_values = pd.DataFrame(
        shap_values, columns=X_train.columns.tolist())

    return {"model_r2":{"train_data":r2_model_train,"test_data":r2_model_test},
        "df_prediction": df_prediction.to_json(), 
            "df_shap_values": df_shap_values.to_json()}