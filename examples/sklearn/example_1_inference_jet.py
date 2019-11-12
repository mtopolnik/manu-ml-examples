import pandas as pd
import pickle
import json
import yaml


def read_csv(filename, has_header=True):
    df = pd.read_csv(filename, header=0 if has_header else None)
    df.columns = [col.strip() for col in df.columns]
    return df


def read_coll(filename):
    with open(filename, 'r') as f:
        return json.load(f)


inputs_folder = 'ai_inputs'
ai_folder = 'outputs1'
labels_dict = read_coll(ai_folder + '/labels_dict.json')
inv_labels_dict = {v: k for k, v in labels_dict.items()}
cols = read_coll(ai_folder + '/input_column_names.json')
occupation_stats = pd.read_csv(inputs_folder + '/occupation_stats.csv')
feature_selected_columns = read_coll(ai_folder + '/selected_features.txt')
inter_1 = pd.read_csv(ai_folder + '/education_mean.csv')
with open(inputs_folder + '/ai-config.yml', 'r') as stream:
    config = yaml.load(stream)
hours_info = read_coll(ai_folder + '/hours_info.json')
with open(ai_folder + '/onehot.model', 'rb') as f:
    encoder = pickle.load(f)
with open(ai_folder + '/rf.model', 'rb') as f:
    model = pickle.load(f)


def handle(input_json):
    row_dict = json.loads(input_json)
    df = pd.DataFrame([row_dict])
    df = df.merge(occupation_stats, how='inner', on='occupation')
    df = df[feature_selected_columns]
    df = df.merge(inter_1, on='education')
    df['age'] = config['age_multiplier'] * df['age']
    df['hours-per-week'] = df['hours-per-week'] - hours_info['min']
    df['hours-per-week'] = df['hours-per-week'] / (hours_info['max'] - hours_info['min'])
    x_test_encoded = encoder.transform(df)
    predicted_probability = model.predict_proba(x_test_encoded).transpose()[1][0]
    predicted_label = model.predict(x_test_encoded).transpose()[0]
    result = {
        'probability': float(predicted_probability),
        'label': inv_labels_dict[int(predicted_label)]
    }
    return json.dumps(result)
