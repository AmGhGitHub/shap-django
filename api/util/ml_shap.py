from ast import Lambda
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

def round_df(input_df, n_decimial):
    df=input_df.copy()
    return df.applymap(lambda x:round(x,n_decimial))


def generate_ml_and_shap_data(df_js):
    df = pd.DataFrame(json.loads(df_js))

    output_column_name = 'output'
    X = df.loc[:, df.columns != output_column_name]
    y = df.iloc[:, df.columns == output_column_name]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    xgb_reg = XGBRegressor(n_estimators=20)
    xgb_reg.fit(X_train, y_train)
    y_train_pred = xgb_reg.predict(X_train)
    y_test_pred = xgb_reg.predict(X_test)
    
    model_r2_train_data =round(r2_score(y_train, y_train_pred),3)
    model_r2_test_data = round(r2_score(y_test, y_test_pred),3)


    df_train_pred_sample =round_df(get_sample_def(
        y_train[output_column_name].values, y_train_pred, 0.2),4)
    
    df_test_pred_sample =round_df( get_sample_def(
        y_test[output_column_name].values, y_test_pred, 0.3),4)
    
    model_pred_train_data=df_train_pred_sample.values.tolist()
    model_pred_test_data=df_test_pred_sample.values.tolist()
    # print(model_test_pred)

    explainer = shap.TreeExplainer(xgb_reg)
    shap_values = explainer.shap_values(X_train)

    df_train_shapValues = pd.DataFrame(
        shap_values, columns=X_train.columns.tolist())
        
    df_train_shapValues_sample=round_df(df_train_shapValues.sample(frac=0.4,replace=False),4)

    shap_features=[col for col in df_train_shapValues.columns]
    # shap_values_sample=[df_train_shapValues_sample[col].values.tolist() for col in df_train_shapValues_sample.columns]
    shap_values_sample=[np.c_[df_train_shapValues_sample[col].values,np.ones(len(df_train_shapValues_sample[col]))+(i-1)
                                ].tolist() for i, col in enumerate(df_train_shapValues_sample.columns)]
    # shap_values_sample_3=[val for val in shap_values_sample_2[0]]
    # shap_values_sample_2=[[i, val] for i, val in enumerate(df_train_shapValues_sample[col].values for col in df_train_shapValues_sample.columns)]
    
    feature_importance=np.abs(shap_values).mean(0).tolist()
    
    return {"model":{"r2":{"train_data":model_r2_train_data,"test_data":model_r2_test_data},
                     "prediction":{"train_data":model_pred_train_data,"test_data":model_pred_test_data}
                     },
            "shap":{"features":shap_features, 
                    "sample_values":shap_values_sample,
                    "feature_importance":feature_importance
            }
            }
