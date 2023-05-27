from fastapi import FastAPI
from typing import List
from model_backend import OutputModelo, InputModelo, APIBackend

app = FastAPI(title="API para clasificación de géneros de películas" , version="0.0.1")
'''API desarrollada para la asignatura de MIAD'''

@app.post("/predict_probas", response_model=List[OutputModelo],
          tags=["Proyecto 2: Taggeo de género de películas en función de la review"])
async def predict_probas(inputs:List[InputModelo] ):
    '''Endpoint de la API que se encarga de hacer la predicción de la(s) probabilidad(es) de cada género dado para uno o varios reviews'''
    response = list()

    for Input in inputs:
        model = APIBackend(Input.review)
        response.append(model.predict_proba()[0])

    return response

