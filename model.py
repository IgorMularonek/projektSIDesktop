import pickle

def load_model(model_name):
    with open(f"{model_name}_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

def load_label_encoders():
    with open("label_encoders.pkl", "rb") as f:
        label_encoders = pickle.load(f)
    return label_encoders

def make_prediction(model, encoders, data):
    data_encoded = []
    for col, value in data.items():
        if col in encoders:
            data_encoded.append(encoders[col].transform([value])[0])
        else:
            data_encoded.append(value)
    prediction = model.predict([data_encoded])
    return "Zadowolony" if prediction[0] == 1 else "Niezadowolony"
