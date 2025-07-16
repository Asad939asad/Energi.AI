# import joblib
# import numpy as np





# import numpy as np
# import tensorflow as tf
# from tensorflow.keras import layers, models, Input
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint































# import numpy as np
# import tensorflow as tf
# from tensorflow.keras import layers, models, Input
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint



















# import numpy as np
# import tensorflow as tf
# from tensorflow.keras import layers, models, Input
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# # === STEP 1: Convert list-based data to NumPy arrays ===
# # X_array = np.array(X)  # X is a list of sequences
# # y_array = np.array(y)  # y is a list of scalar targets (solar production)

# # # === STEP 2: Split into train and validation sets ===
# # X_train, X_val, y_train, y_val = train_test_split(
# #     X_array, y_array, test_size=0.05, random_state=42
# # )

# # === STEP 3: Custom Attention Layer ===
# class AttentionLayer(tf.keras.layers.Layer):
#     def __init__(self):
#         super(AttentionLayer, self).__init__()

#     def call(self, encoder_outputs, decoder_hidden):
#         # encoder_outputs: (batch, seq_len, hidden_dim)
#         # decoder_hidden: (batch, hidden_dim)
#         score = tf.matmul(encoder_outputs, tf.expand_dims(decoder_hidden, axis=-1))  # (batch, seq_len, 1)
#         score = tf.squeeze(score, axis=-1)  # (batch, seq_len)
#         attention_weights = tf.nn.softmax(score, axis=1)  # (batch, seq_len)
#         context_vector = tf.reduce_sum(
#             encoder_outputs * tf.expand_dims(attention_weights, -1), axis=1
#         )  # (batch, hidden_dim)
#         return context_vector, attention_weights

# # === STEP 4: Build Model ===
# def build_lstm_attention_model(input_shape, lstm_units=128, dropout_rate=0.4):
#     inputs = Input(shape=input_shape)  # (seq_len, num_features)
#     x = inputs

#     for i in range(20):
#         x = layers.LSTM(lstm_units, return_sequences=True, name=f'lstm_{i+1}')(x)
#         x = layers.LayerNormalization()(x)
#         x = layers.Dropout(dropout_rate)(x)

#     attention = AttentionLayer()
#     context_vector, _ = attention(x, x[:, -1, :])  # decoder_hidden = last timestep output

#     x = layers.Dense(64, activation='relu')(context_vector)
#     x = layers.Dropout(dropout_rate)(x)
#     outputs = layers.Dense(1)(x)  # Predict solar production

#     model = models.Model(inputs=inputs, outputs=outputs)
#     return model

# # === STEP 5: Build and Compile ===
# input_shape = (2, 9)  # (seq_len, num_features)
# model1 = build_lstm_attention_model(input_shape)
# model1.compile(optimizer='adam', loss='mse', metrics=['mae'])

# # model.summary()


























# import numpy as np
# import tensorflow as tf
# from tensorflow.keras import layers, models, Input
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint




# # === STEP 3: Custom Attention Layer ===
# class AttentionLayer(tf.keras.layers.Layer):
#     def __init__(self):
#         super(AttentionLayer, self).__init__()

#     def call(self, encoder_outputs, decoder_hidden):
#         # encoder_outputs: (batch, seq_len, hidden_dim)
#         # decoder_hidden: (batch, hidden_dim)
#         score = tf.matmul(encoder_outputs, tf.expand_dims(decoder_hidden, axis=-1))  # (batch, seq_len, 1)
#         score = tf.squeeze(score, axis=-1)  # (batch, seq_len)
#         attention_weights = tf.nn.softmax(score, axis=1)  # (batch, seq_len)
#         context_vector = tf.reduce_sum(
#             encoder_outputs * tf.expand_dims(attention_weights, -1), axis=1
#         )  # (batch, hidden_dim)
#         return context_vector, attention_weights

# # === STEP 4: Build Model ===
# def build_lstm_attention_model(input_shape, lstm_units=128, dropout_rate=0.4):
#     inputs = Input(shape=input_shape)  # (seq_len, num_features)
#     x = inputs

#     for i in range(10):
#         x = layers.LSTM(lstm_units, return_sequences=True, name=f'lstm_{i+1}')(x)
#         x = layers.LayerNormalization()(x)
#         x = layers.Dropout(dropout_rate)(x)

#     attention = AttentionLayer()
#     context_vector, _ = attention(x, x[:, -1, :])  # decoder_hidden = last timestep output

#     x = layers.Dense(64, activation='relu')(context_vector)
#     x = layers.Dropout(dropout_rate)(x)
#     outputs = layers.Dense(1)(x)  # Predict solar production

#     model = models.Model(inputs=inputs, outputs=outputs)
#     return model

# # === STEP 5: Build and Compile ===
# input_shape = (10, 12)  # (seq_len, num_features)
# model2 = build_lstm_attention_model(input_shape)
# model2.compile(optimizer='adam', loss='mse', metrics=['mae'])

# # model2.summary()




























# X_seq_ = [[[0.05504587, 0.        , 0.4729064 , 0.0029703 , 0.28860759,
#          0.73563218, 0.        , 0.        , 0.17647059],
#         [0.05504587, 0.        , 0.47906404, 0.00308031, 0.26329114,
#          0.7816092 , 0.        , 0.        , 0.23529412]]]
















# X_seq = [[[8.61463972e-02, 7.56769710e-01, 5.12979947e-03, 5.67499802e-05,
#          7.90607582e-05, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
#          2.00000000e-01, 5.00000000e-01, 1.00000000e+00, 0.00000000e+00],
#         [8.33990697e-02, 7.56769710e-01, 4.66345406e-03, 4.98711947e-05,
#          9.50865876e-05, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
#          4.00000000e-01, 5.00000000e-01, 1.00000000e+00, 0.00000000e+00],
#         [7.81764274e-02, 7.62760604e-01, 5.12979947e-03, 7.48067921e-05,
#          9.08130331e-05, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
#          6.00000000e-01, 2.50000000e-01, 1.00000000e+00, 0.00000000e+00],
#         [7.27361750e-02, 7.71147855e-01, 4.81890253e-03, 3.78333201e-05,
#          7.90607582e-05, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
#          8.00000000e-01, 2.50000000e-01, 1.00000000e+00, 0.00000000e+00],
#         [7.08864891e-02, 7.85526000e-01, 4.81890253e-03, 4.72916502e-05,
#          1.03633697e-04, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
#          1.00000000e+00, 2.50000000e-01, 1.00000000e+00, 0.00000000e+00],
#         [6.51198216e-02, 7.95111431e-01, 4.66345406e-03, 3.78333201e-05,
#          9.08130331e-05, 0.00000000e+00, 0.00000000e+00, 4.34782609e-02,
#          0.00000000e+00, 2.50000000e-01, 1.00000000e+00, 0.00000000e+00],
#         [6.11756385e-02, 8.01102324e-01, 5.44069641e-03, 4.38522574e-05,
#          8.76078672e-05, 0.00000000e+00, 0.00000000e+00, 4.34782609e-02,
#          2.00000000e-01, 2.50000000e-01, 1.00000000e+00, 0.00000000e+00],
#         [6.61262683e-02, 7.99904146e-01, 4.81890253e-03, 5.33105875e-05,
#          1.38890521e-04, 0.00000000e+00, 0.00000000e+00, 4.34782609e-02,
#          4.00000000e-01, 2.50000000e-01, 1.00000000e+00, 0.00000000e+00],
#         [6.10396322e-02, 7.90318716e-01, 4.97435100e-03, 4.98711947e-05,
#          1.06838862e-04, 0.00000000e+00, 0.00000000e+00, 4.34782609e-02,
#          6.00000000e-01, 2.50000000e-01, 1.00000000e+00, 0.00000000e+00],
#         [6.17196638e-02, 7.92715073e-01, 4.81890253e-03, 4.04128647e-05,
#          1.03633697e-04, 0.00000000e+00, 0.00000000e+00, 4.34782609e-02,
#          8.00000000e-01, 2.50000000e-01, 1.00000000e+00, 0.00000000e+00]]]




















# # Load the scalers
# model1.load_weights("main_files/best_model_123weights.weights.h5")
# model2.load_weights("main_files/Total_Power.weights.h5")

# # Load the scalers for Y (target: total_power) and Y_ (target: solar production)
# scalerY = joblib.load('main_files/scalerY.save') #Power
# scalerY_ = joblib.load('main_files/scalerY_.save') #Solar Production

# X_seq_array_ = np.array(X_seq) # Convert to numpy array
# # Make prediction for total_power (normalized)
# normalized_pred_power = model2.predict(X_seq_array_)  # shape: (1, 1)
# # Inverse transform to get actual power (float)
# actual_pred_power = scalerY.inverse_transform(normalized_pred_power)[0][0]  # single scalar value


# # Make prediction for solar production (normalized)
# X_seq_array = np.array(X_seq_) # Convert to numpy array
# normalized_pred_solar = model1.predict(X_seq_array)  # shape: (1, 1)
# # Inverse transform to get actual solar production (float)
# actual_pred_solar = scalerY_.inverse_transform(normalized_pred_solar)[0][0]  # single scalar value


# row = {
#     'Temperature': 28,
#     'Humidity': 75,
#     'WindSpeed': 3.0,
#     'GeneralDiffuseFlows': 0.2,
#     'DiffuseFlows': 0.1,
#     'Radiation': 620,
#     'RelativeAirHumidity': 85,
#     'total_power': 450,  # true value (for comparison)
#     'predicted_total_power': round(actual_pred_power.item(), 1),  # convert np.float32 → float
#     'hour': 10,
#     'minute': 30,
#     'season': 'summer',
#     'is_peak_hour': True,
#     'day': 10,
#     'month': 7,
#     'AirPressure': 1010,
#     'Sunshine': 7.5,
#     'AirTemperature': 29,
#     'SystemProduction': round(actual_pred_solar.item(), 1)  # convert np.float32 → float
# }

# if row['hour'] <= 12 and row['hour'] > 0:
#     Zone = "AM"
# else:
#     row['hour'] = row['hour'] - 12
#     Zone = "PM"

# row['Zone'] = Zone
# # Calculate emissions
# emissions = row['predicted_total_power'] * 0.425
# row['CO2'] = round(emissions, 1)  # Already a float, no need for .item()
# # Display result
# # print(f"Predicted Total Power (Actual Value): {actual_pred_power:.1f}")
# # print(f"Predicted Solar Production (Actual Value): {actual_pred_solar:.1f}")
# print(row)





def predict_row():
    import joblib
    import numpy as np

    import numpy as np
    import tensorflow as tf
    from tensorflow.keras import layers, models, Input
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

    import numpy as np
    import tensorflow as tf
    from tensorflow.keras import layers, models, Input
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

    import numpy as np
    import tensorflow as tf
    from tensorflow.keras import layers, models, Input
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

    class AttentionLayer(tf.keras.layers.Layer):
        def __init__(self):
            super(AttentionLayer, self).__init__()

        def call(self, encoder_outputs, decoder_hidden):
            score = tf.matmul(encoder_outputs, tf.expand_dims(decoder_hidden, axis=-1))
            score = tf.squeeze(score, axis=-1)
            attention_weights = tf.nn.softmax(score, axis=1)
            context_vector = tf.reduce_sum(
                encoder_outputs * tf.expand_dims(attention_weights, -1), axis=1
            )
            return context_vector, attention_weights

    def build_lstm_attention_model(input_shape, lstm_units=128, dropout_rate=0.4):
        inputs = Input(shape=input_shape)
        x = inputs

        for i in range(20):
            x = layers.LSTM(lstm_units, return_sequences=True, name=f'lstm_{i+1}')(x)
            x = layers.LayerNormalization()(x)
            x = layers.Dropout(dropout_rate)(x)

        attention = AttentionLayer()
        context_vector, _ = attention(x, x[:, -1, :])

        x = layers.Dense(64, activation='relu')(context_vector)
        x = layers.Dropout(dropout_rate)(x)
        outputs = layers.Dense(1)(x)

        model = models.Model(inputs=inputs, outputs=outputs)
        return model

    input_shape = (2, 9)
    model1 = build_lstm_attention_model(input_shape)
    model1.compile(optimizer='adam', loss='mse', metrics=['mae'])

    class AttentionLayer(tf.keras.layers.Layer):
        def __init__(self):
            super(AttentionLayer, self).__init__()

        def call(self, encoder_outputs, decoder_hidden):
            score = tf.matmul(encoder_outputs, tf.expand_dims(decoder_hidden, axis=-1))
            score = tf.squeeze(score, axis=-1)
            attention_weights = tf.nn.softmax(score, axis=1)
            context_vector = tf.reduce_sum(
                encoder_outputs * tf.expand_dims(attention_weights, -1), axis=1
            )
            return context_vector, attention_weights

    def build_lstm_attention_model(input_shape, lstm_units=128, dropout_rate=0.4):
        inputs = Input(shape=input_shape)
        x = inputs

        for i in range(10):
            x = layers.LSTM(lstm_units, return_sequences=True, name=f'lstm_{i+1}')(x)
            x = layers.LayerNormalization()(x)
            x = layers.Dropout(dropout_rate)(x)

        attention = AttentionLayer()
        context_vector, _ = attention(x, x[:, -1, :])

        x = layers.Dense(64, activation='relu')(context_vector)
        x = layers.Dropout(dropout_rate)(x)
        outputs = layers.Dense(1)(x)

        model = models.Model(inputs=inputs, outputs=outputs)
        return model

    input_shape = (10, 12)
    model2 = build_lstm_attention_model(input_shape)
    model2.compile(optimizer='adam', loss='mse', metrics=['mae'])

    X_seq_ = [[[0.05504587, 0.        , 0.4729064 , 0.0029703 , 0.28860759,
                0.73563218, 0.        , 0.        , 0.17647059],
               [0.05504587, 0.        , 0.47906404, 0.00308031, 0.26329114,
                0.7816092 , 0.        , 0.        , 0.23529412]]]

    X_seq = [[[8.61463972e-02, 7.56769710e-01, 5.12979947e-03, 5.67499802e-05,
               7.90607582e-05, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
               2.00000000e-01, 5.00000000e-01, 1.00000000e+00, 0.00000000e+00],
              [8.33990697e-02, 7.56769710e-01, 4.66345406e-03, 4.98711947e-05,
               9.50865876e-05, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
               4.00000000e-01, 5.00000000e-01, 1.00000000e+00, 0.00000000e+00],
              [7.81764274e-02, 7.62760604e-01, 5.12979947e-03, 7.48067921e-05,
               9.08130331e-05, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
               6.00000000e-01, 2.50000000e-01, 1.00000000e+00, 0.00000000e+00],
              [7.27361750e-02, 7.71147855e-01, 4.81890253e-03, 3.78333201e-05,
               7.90607582e-05, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
               8.00000000e-01, 2.50000000e-01, 1.00000000e+00, 0.00000000e+00],
              [7.08864891e-02, 7.85526000e-01, 4.81890253e-03, 4.72916502e-05,
               1.03633697e-04, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
               1.00000000e+00, 2.50000000e-01, 1.00000000e+00, 0.00000000e+00],
              [6.51198216e-02, 7.95111431e-01, 4.66345406e-03, 3.78333201e-05,
               9.08130331e-05, 0.00000000e+00, 0.00000000e+00, 4.34782609e-02,
               0.00000000e+00, 2.50000000e-01, 1.00000000e+00, 0.00000000e+00],
              [6.11756385e-02, 8.01102324e-01, 5.44069641e-03, 4.38522574e-05,
               8.76078672e-05, 0.00000000e+00, 0.00000000e+00, 4.34782609e-02,
               2.00000000e-01, 2.50000000e-01, 1.00000000e+00, 0.00000000e+00],
              [6.61262683e-02, 7.99904146e-01, 4.81890253e-03, 5.33105875e-05,
               1.38890521e-04, 0.00000000e+00, 0.00000000e+00, 4.34782609e-02,
               4.00000000e-01, 2.50000000e-01, 1.00000000e+00, 0.00000000e+00],
              [6.10396322e-02, 7.90318716e-01, 4.97435100e-03, 4.98711947e-05,
               1.06838862e-04, 0.00000000e+00, 0.00000000e+00, 4.34782609e-02,
               6.00000000e-01, 2.50000000e-01, 1.00000000e+00, 0.00000000e+00],
              [6.17196638e-02, 7.92715073e-01, 4.81890253e-03, 4.04128647e-05,
               1.03633697e-04, 0.00000000e+00, 0.00000000e+00, 4.34782609e-02,
               8.00000000e-01, 2.50000000e-01, 1.00000000e+00, 0.00000000e+00]]]

    model1.load_weights("main_files/best_model_123weights.weights.h5")
    model2.load_weights("main_files/Total_Power.weights.h5")

    scalerY = joblib.load('main_files/scalerY.save')
    scalerY_ = joblib.load('main_files/scalerY_.save')

    X_seq_array_ = np.array(X_seq)
    normalized_pred_power = model2.predict(X_seq_array_)
    actual_pred_power = scalerY.inverse_transform(normalized_pred_power)[0][0]

    X_seq_array = np.array(X_seq_)
    normalized_pred_solar = model1.predict(X_seq_array)
    actual_pred_solar = scalerY_.inverse_transform(normalized_pred_solar)[0][0]

    row = {
        'Temperature': 8,
        'Humidity': 20,
        'WindSpeed': 10.0,
        'GeneralDiffuseFlows': 0.2,
        'DiffuseFlows': 0.1,
        'Radiation': 620,
        'RelativeAirHumidity': 85,
        'total_power': 450,
        'predicted_total_power': round(actual_pred_power.item(), 1),
        'hour': 23,
        'minute': 30,
        'season': 'Autumn',
        'is_peak_hour': 'False',
        'day': 10,
        'month': 7,
        'AirPressure': 1010,
        'Sunshine': 6.5,
        'AirTemperature': 29,
        'SystemProduction': round(actual_pred_solar.item(), 1)
    }

    if row['hour'] <= 12 and row['hour'] > 0:
        Zone = "AM"
    else:
        row['hour'] = row['hour'] - 12
        Zone = "PM"

    row['Zone'] = Zone
    emissions = row['predicted_total_power'] * 0.425
    row['CO2'] = round(emissions, 1)
    return row
