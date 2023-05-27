from pydantic import BaseModel as BM
from pydantic import Field
import joblib
import pandas as pd
import re
from nltk.corpus import stopwords

import nltk
nltk.download('stopwords')

class InputModelo(BM):
    """
      Clase que define las entradas del modelo según las verá el usuario.
    """
    review: str = Field(max_length=5000, min_length=5)

    class Config:
        schema_extra = {
            "example": {
                "review": "in los angeles , the editor of a publishing house carol hunnicut goes to a blind date with the lawyer michael tarlow , who has embezzled the powerful mobster leo watts . carol accidentally witnesses the murder of michel by leo ' s hitman . the scared carol sneaks out of michael ' s room and hides in an isolated cabin in canada . meanwhile the deputy district attorney robert caulfield and sgt . dominick benti discover that carol is a witness of the murder and they report the information to caulfield ' s chief martin larner and they head by helicopter to canada to convince carol to testify against leo . however they are followed and the pilot and benti are murdered by the mafia . caulfield and carol flees and they take a train to vancouver . caulfield hides carol in his cabin and he discloses that there are three hitman in the train trying to find carol and kill her . but they do not know her and caulfield does not know who might be the third killer from the mafia and who has betrayed him in his office ."
            }
        }


class OutputModelo(BM):
    """
     Clase que define la salida del modelo según la verá el usuario.
     """
    p_Action: float =Field(ge=0, le=1, description="Probabilidad de que la review dada sea de acción")
    p_Adventure: float =Field(ge=0, le=1)
    p_Animation:float =Field(ge=0, le=1)
    p_Biography: float =Field(ge=0, le=1)
    p_Comedy: float =Field(ge=0, le=1)
    p_Crime: float =Field(ge=0, le=1)
    p_Documentary: float =Field(ge=0, le=1)
    p_Drama: float =Field(ge=0, le=1)
    p_Family: float =Field(ge=0, le=1)
    p_Fantasy: float =Field(ge=0, le=1)
    p_Film_Noir:float =Field(ge=0, le=1)
    p_History: float =Field(ge=0, le=1)
    p_Horror: float =Field(ge=0, le=1)
    p_Music: float =Field(ge=0, le=1)
    p_Musical: float =Field(ge=0, le=1)
    p_Mystery: float =Field(ge=0, le=1)
    p_News: float =Field(ge=0, le=1)
    p_Romance: float =Field(ge=0, le=1)
    p_Sci_Fi: float =Field(ge=0, le=1)
    p_Short: float =Field(ge=0, le=1)
    p_Sport:float =Field(ge=0, le=1)
    p_Thriller: float =Field(ge=0, le=1)
    p_War: float =Field(ge=0, le=1)
    p_Western:float =Field(ge=0, le=1)
    class Config:
        schema_extra = {
            "example":   {
    "p_Action": 0.0014689359382014859,
    "p_Adventure": 8.566928222795404e-8,
    "p_Animation": 2.574882483672089e-11,
    "p_Biography": 5.497779700865337e-11,
    "p_Comedy": 6.560188784965996e-7,
    "p_Crime": 0.1372898855169001,
    "p_Documentary": 1.0097643293377614e-12,
    "p_Drama": 0.000010000338750511898,
    "p_Family": 6.861784740259703e-11,
    "p_Fantasy": 6.917978138105743e-8,
    "p_Film_Noir": 0.0023248609733912233,
    "p_History": 1.6366707523491114e-9,
    "p_Horror": 0.00035679823254913324,
    "p_Music": 9.213671396137869e-13,
    "p_Musical": 2.045806982026245e-10,
    "p_Mystery": 0.3707154256969401,
    "p_News": 3.606805376531604e-7,
    "p_Romance": 4.194402555121554e-8,
    "p_Sci_Fi": 0.0000052136192525991465,
    "p_Short": 5.240854521483309e-9,
    "p_Sport": 1.9694361250752544e-12,
    "p_Thriller": 0.4878276530700981,
    "p_War": 2.8026025453762033e-10,
    "p_Western": 5.605803726065324e-9
  }
        }

sw = stopwords.words('english')

def pre_process(df, col):
    df['x'] = df[col].apply(
        lambda x: ' '.join([y for y in re.sub("[^a-zA-Z0-9 ]", '', x).split(' ') if y not in sw]))
    return df

class APIBackend:
    def __init__(self, review):
        self.review = review
    def _cargar_modelo(self, model_name="NaiveBayes3.pkl", vect_name ="CountVect.pkl"):
        self.model = joblib.load(model_name)
        self.vect = joblib.load(vect_name)
    def _preparar_dato(self):
        review = self.review
        vect = self.vect
        data_prediction = pd.DataFrame(columns =['review'], data =[[review]])
        data_prediction = pre_process(data_prediction, "review")
        X_test = vect.transform(data_prediction['x'])
        return X_test


    def predict_proba(self):
        self._cargar_modelo()
        x = self._preparar_dato()
        cols = ['p_Action', 'p_Adventure', 'p_Animation', 'p_Biography', 'p_Comedy', 'p_Crime', 'p_Documentary',
                'p_Drama', 'p_Family',
                'p_Fantasy', 'p_Film_Noir', 'p_History', 'p_Horror', 'p_Music', 'p_Musical', 'p_Mystery', 'p_News',
                'p_Romance',
                'p_Sci_Fi', 'p_Short', 'p_Sport', 'p_Thriller', 'p_War', 'p_Western']
        prediction = self.model.predict_proba(x)
        res = pd.DataFrame(prediction, index=[self.review], columns=cols)
        return res.to_dict(orient="records")

