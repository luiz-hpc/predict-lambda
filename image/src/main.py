import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

def forecast_RW_fct(da,pred=1):
    forecast = np.zeros(shape=(pred,da.shape[1]))
    for i in range(0,da.shape[1]):
        forecast[:,i] = np.ones(shape=(pred))*da[da.shape[0]-1,i]
    return forecast

scaler = MinMaxScaler(feature_range=(0, 1))

def handler(event, _):
    print(event)
    query_params = event.get("queryStringParameters", {})

    step = query_params.get("step") if query_params else 1
    
    step = int(event.get("step", step))

    if step < 1:
        raise ValueError("Step menor que 1")
    
    print(f"Running predictions for step={step}")

    forecast_step = step # how many *day* to forecast forward

    y_full_df = pd.read_csv('s3://lstm-rw-bucket/DI1_settle.csv')
    y_full = y_full_df.to_numpy()

    matu = np.array([[1/12, 3/12, 6/12, 1, 1.5, 2, 3, 5, 10]])
    y = y_full

    y = np.delete(y, 0, 1)
    y = np.array(y, dtype=float)

    data = y

    # PRODUCE FORECASTS
    forecast_RW = forecast_RW_fct(data,forecast_step) #Random walk

    # LSTM

    # Normalizar os dados
    scaled_data = scaler.fit_transform(y)

    look_back = 10

    def create_prediction_dataset(dataset, look_back=1):
        dataX = []
        for i in range(look_back):
            a = dataset[(-look_back+i):, :]
            dataX.append([a])
        return dataX

    predictX = create_prediction_dataset(scaled_data, look_back)

    # load model

    model = load_model('model.keras')

    forecastPredict = []

    for i in range(forecast_step):

        validX = predictX[i][0]  # Access the array inside the list
        if len(forecastPredict) > 0:
            validX = np.vstack([validX, forecastPredict[-1]])  # Stack arrays vertically
        validX = np.reshape(validX, (1, validX.shape[0], validX.shape[1]))  # Reshape to match expected input shape
        prediction = model.predict(validX, verbose=0)

        forecastPredict.append(prediction[0])

    # Convert list to numpy array
    forecastPredict = np.array(forecastPredict)

    # Inverter a previsão
    forecastPredict = scaler.inverse_transform(forecastPredict)[-1]

    data = {
        "RW": forecast_RW[-1, :],
        "LSTM": forecastPredict.flatten()
    }

    df = pd.DataFrame(data, index=matu.flatten())

    return {
        "statusCode": 200,
        "body": {"data": df.reset_index().to_dict(orient='records')},
    }