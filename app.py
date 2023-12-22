from flask import Flask, request, render_template, make_response, jsonify
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
df = pd.read_csv('./dataset/Dataset maps - Dataset.csv')
df.dropna(inplace=True)
scaler = StandardScaler()
df[['T_LATITUDE', 'T_LONGITUDE', 'T_Dekat_Sekolah', 'T_Dekat_Rumah_Sakit', 'T_Dekat_Pasar', 'T_Dekat_Rumah_Warga',
    'T_Fasilitas_Umum_Masyarakat',
    'T_Dilalui_Kendaraan_Umum', 'T_Kepadatan_Jalan', 'T_Dekat_Penginapan_Kost', 'T_Dekat_PT', 'T_Dekat_Rumah_Ibadah',
    'T_Dekat_Stasiun', 'T_Dekat_Bandara',
    'T_Dekat_Terminal']] = scaler.fit_transform(
    df[['LATITUDE', 'LONGITUDE', 'Dekat_Sekolah', 'Dekat_Rumah_Sakit', 'Dekat_Pasar', 'Dekat_Rumah_Warga',
        'Fasilitas_Umum_Masyarakat', 'Dilalui_Kendaraan_Umum', 'Kepadatan_Jalan', 'Dekat_Penginapan_Kost', 'Dekat_PT',
        'Dekat_Rumah_Ibadah',
        'Dekat_Stasiun', 'Dekat_Bandara', 'Dekat_Terminal']])

with open('model_1.sav', 'rb') as f:
    loaded_model = pickle.load(f)
scaler = StandardScaler()
scaler.fit(df[['Dekat_Sekolah', 'Dekat_Rumah_Sakit', 'Dekat_Pasar', 'Dekat_Rumah_Warga',
               'Fasilitas_Umum_Masyarakat', 'Dilalui_Kendaraan_Umum', 'Kepadatan_Jalan', 'Dekat_Penginapan_Kost',
               'Dekat_PT', 'Dekat_Rumah_Ibadah', 'Dekat_Stasiun', 'Dekat_Bandara', 'Dekat_Terminal']])


# Fungsi hitung jarak
def euclidean_distance(lat1, lon1, lat2, lon2):
    return np.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2)


def find_nearest_points(user_lat, user_lon, dataframe, user_cluster):
    # buat dataset baru
    cluster_df = dataframe[dataframe['kmeans_5'] == user_cluster].copy()
    # masukan data ke kolom distance dengan menghitung jarak
    cluster_df['distance'] = cluster_df.apply(
        lambda row: euclidean_distance(user_lat, user_lon, row['LATITUDE'], row['LONGITUDE']),
        axis=1
    )

    # Urut 5 terdekat
    nearest_points = cluster_df.nsmallest(5, 'distance')
    result_array = nearest_points[['LATITUDE', 'LONGITUDE']].to_numpy()

    return result_array


with open('model_1.sav', 'rb') as f:
    loaded_model = pickle.load(f)
loaded_model.fit(df[['T_Dekat_Sekolah', 'T_Dekat_Rumah_Sakit', 'T_Dekat_Pasar', 'T_Dekat_Rumah_Warga',
                     'T_Fasilitas_Umum_Masyarakat', 'T_Dilalui_Kendaraan_Umum', 'T_Kepadatan_Jalan',
                     'T_Dekat_Penginapan_Kost', 'T_Dekat_PT', 'T_Dekat_Rumah_Ibadah', 'T_Dekat_Stasiun',
                     'T_Dekat_Bandara', 'T_Dekat_Terminal']])
df['kmeans_5'] = loaded_model.labels_


@app.route('/', methods=['GET'])
def hello_world():  # put application's code here
    return render_template('index.html')


@app.route('/read-data', methods=['POST'])
def read_data():
    if (request.method == 'POST'):
        data = request.form

        return jsonify({
            "status": {
                "code": 200,
                "message": "Success recomendation"
            },
            "data": {
                'latitude': data['latitude'],
                'longitude': data['longitude'],
                'industry': data['industry'],
            }
        }), 200


@app.route('/predict', methods=['POST'])
def predict():
    if (request.method == 'POST'):
        data = request.get_json(force=True)
        latitude = float(data['latitude'])
        longitude = float(data['longitude'])
        industry = int(data['industry'])
        nearest_points = find_nearest_points(latitude, longitude, df, user_cluster=industry).tolist()
        return jsonify({
            "status": {
                "code": 200,
                "message": "Sukses mencari rekomendasi"
            },
            "data": {
                "titik": list(nearest_points)
            }
        }), 200


@app.route('/input-tempat', methods=['GET'])
def inputTempat():
    return render_template('input.html')


@app.route('/predictloc', methods=['POST'])
def predictloc():
    if (request.method == 'POST'):
        data = request.get_json(force=True)
        user_data = {
            'Dekat_Sekolah': int(data['dekatSekolah']),
            'Dekat_Rumah_Sakit': int(data['dekatRumahSakit']),
            'Dekat_Pasar': int(data['dekatPasar']),
            'Dekat_Rumah_Warga': int(data['dekatRumahWarga']),
            'Fasilitas_Umum_Masyarakat': int(data['dekatFasum']),
            'Dilalui_Kendaraan_Umum': int(data['dilaluiKendaraanUmum']),
            'Kepadatan_Jalan': int(data['kepadatanJalan']),
            'Dekat_Penginapan_Kost': int(data['dekatPenginapan']),
            'Dekat_PT': int(data['dekatPT']),
            'Dekat_Rumah_Ibadah': int(data['dekatRumahIbadah']),
            'Dekat_Stasiun': int(data['dekatStasiun']),
            'Dekat_Bandara': int(data['dekatBandara']),
            'Dekat_Terminal': int(data['dekatTerminal']),
        }

    new_data_scaled = scaler.transform(np.array(list(user_data.values())).reshape(1, -1)).tolist()


    # Prediksi cluster untuk data baru
    predicted_cluster = loaded_model.predict(new_data_scaled).tolist()[0]

    return jsonify({
        "status": {
            "code": 200,
            "message": "berhasil menampilkan rekomendasi"
        },
        "data": {
            "hasil_rekomendasi": predicted_cluster
        }
    }), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

