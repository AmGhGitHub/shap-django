import json
import math
import numpy as np
import matplotlib.pyplot as plt
import io

from celery import current_app
from rest_framework.decorators import api_view
from rest_framework.response import Response

from django.http import HttpResponse

from .tasks import gen_results
from .util.ml_shap import generate_ml_and_shap_data


def calc_nruns(val):
    return int(math.pow(10, int(val)))


@api_view(["POST", "GET"])
def generate_data(request):
    if request.method == 'POST':
        req = request.data.dict()

        sample_size = calc_nruns(req["sample_size_exponent"])
        repeated_rows_pct = int(req["repeated_rows_pct"])
        lst_variables = json.loads(req["variables_data"])
        latex_eq = req["latex_equation"]

        celery_task = gen_results.delay(sample_size, lst_variables,
                                        repeated_rows_pct, latex_eq)

        return Response({"celery_task_id": celery_task.id})

    if request.method == "GET":
        celery_task_id = request.GET.get('task_id', '')
        celery_task = current_app.AsyncResult(celery_task_id)

        response_data = {'status': celery_task.status}
        if celery_task.status == 'SUCCESS':
            data = celery_task.get()
            
            response_data['Histogram data'] = data['Histogram data']
            response_data['ML-SHAP data']=data['ML-SHAP data']

        return Response(response_data)
