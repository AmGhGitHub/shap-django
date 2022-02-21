import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
import io
import json


def generate_shap(input_df, col_dependent_var="output"):
    df = input_df.copy()
    X = df.drop([col_dependent_var], axis=1)
    y = df[col_dependent_var]

    model = XGBRegressor(random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)
    score_train = model.score(X_train, y_train)
    score_test = model.score(X_test, y_test)

    y_pred_test = model.predict(X_test)

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_test, check_additivity=True)
    df_XGB = pd.DataFrame(
        data={'Y_Actual': y_test, 'Prediction': y_pred_test}, index=X_test.index)

    df_SHAP = pd.DataFrame(shap_values,
                           index=X_test.index,
                           columns=[f"{col} contr." for col in X_test.columns])
    df_XGB_SHAP = pd.concat([df_XGB, df_SHAP], axis=1)

    df_report = pd.merge(left=X_test,
                         right=df_XGB_SHAP,
                         how='left',
                         left_index=False,
                         right_index=True,
                         left_on=X_test.index,
                         right_on=df_XGB_SHAP.index)
    # return df_report
    return score_train, score_test


def gen_test_shap_plot(df_js):
    df=pd.DataFrame(json.loads(df_js))
    # # train XGBoost model
    output_column_name = 'output'
    # print(output_column_name)
    X = df.loc[:, df.columns != output_column_name]
    y = df.iloc[:, df.columns == output_column_name]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

   
    xgb_reg = XGBRegressor(n_estimators=20)
    xgb_reg.fit(X_train,y_train);
    r2_model_train=r2_score(y_train,xgb_reg.predict(X_train))
    r2_model_test=r2_score(y_test,xgb_reg.predict(X_test))
    print("R2 test:", r2_model_test)
    print("R2 train:", r2_model_train)
    
    # df_prd_training=pd.DataFrame({"actual":y_train,"Prediction":xgb_reg.predict(X_train),"set":["train"]*len(y_train)})
    # df_prd_test=pd.DataFrame({"actual":y_test,"prediction":xgb_reg.predict(X_test),"set":["test"]*len(y_test)})
    # print(df_prd_training)
    # df_prrediction=pd.concat([df_prd_training,df_prd_test], ignore_index=True)
    # print(df_prrediction)
    
    # df_prrediction_json=df_prrediction.to_json()
    # print(df_prrediction_json)
    # compute SHAP values
    # explainer = shap.TreeExplainer(model, X)
    # shap_values = explainer.shap_values(X)
    # print(shap_values)

    # buf = io.BytesIO()
    # f = plt.figure()

    # shap.summary_plot(shap_values, features=X,
    #                   feature_names=X.columns, plot_type='bar', show=False)
    # # shap.summary_plot(shap_values, X, show=False)
    # f.savefig(buf, format='png', bbox_inches='tight')
    # plt.close()
    # buf.seek(0)
    # img = buf.read()
    return df_prrediction_json
