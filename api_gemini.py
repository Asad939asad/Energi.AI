import pandas as pd
import google.generativeai as genai
from model_predictions import predict_row


def get_power_insight_response(predicted_power, current_hour, current_minute, temp=6, humidity=73.8,
                                wind=0.083, gdiff=0.051, ddiff=0.119, day=6, month=12):
    # === Load Dataset ===
    data = predict_row()  # data is a dictionary
    df_ = pd.DataFrame([data])  # wrap in a list to make it a single-row DataFrame
    print(df_.head(1))
    
    
    current_hour = df_['hour'].iloc[0]
    current_minute = df_['minute'].iloc[0]
    temp = df_['Temperature'].iloc[0]
    humidity = df_['Humidity'].iloc[0]
    wind = df_['WindSpeed'].iloc[0]
    gdiff = df_['GeneralDiffuseFlows'].iloc[0]
    ddiff = df_['DiffuseFlows'].iloc[0]
    # temp = df_['AirTemperature'].iloc[0]
  # fallback
    df = pd.read_csv("main_files/powerconsumption.csv")
    df['datetime'] = pd.to_datetime(df['Datetime'], format='%m/%d/%Y %H:%M')
    df['day'] = df_['day']
    df['month'] = df['datetime'].dt.month
    df['hour'] = df_['hour']
    df['minute'] = df_['minute']
    df['total_power'] = (
        df['PowerConsumption_Zone1'] +
        df['PowerConsumption_Zone2'] +
        df['PowerConsumption_Zone3']
    ) / 250

    df.drop(['PowerConsumption_Zone1', 'PowerConsumption_Zone2', 'PowerConsumption_Zone3'], axis=1, inplace=True)

    # === Peak Season Rules ===
    peak_rules = {
        'winter': {'months': [12, 1, 2],  'start': 17, 'end': 21},
        'spring': {'months': [3, 4, 5],   'start': 18, 'end': 22},
        'summer': {'months': [6, 7, 8],   'start': 19, 'end': 23},
        'autumn': {'months': [9, 10, 11], 'start': 18, 'end': 22},
    }

    def get_season(month):
        for season, rule in peak_rules.items():
            if month in rule['months']:
                return season
        return "unknown"

    def is_peak_hour(hour, season):
        rule = peak_rules.get(season)
        return rule['start'] <= hour < rule['end'] if rule else False

    def get_peak_intensity(value):
        q = df['total_power'].quantile([0.2, 0.4, 0.6, 0.8])
        if value <= q[0.2]: return 1
        elif value <= q[0.4]: return 2
        elif value <= q[0.6]: return 3
        elif value <= q[0.8]: return 4
        else: return 5

    # === Row for prediction ===
    row = {
        'Temperature': 6,
        'Humidity': 73.8,
        'WindSpeed': 0.083,
        'GeneralDiffuseFlows': 0.051,
        'DiffuseFlows': 0.119,
        'day': 6,
        'month': 6,
        'hour': 10,
        'minute': 50,
        'predicted_total_power': 362.27692  # season will be added dynamically
    }

    row['season'] =df_['season'].iloc[0] if 'season' in df_ else "summer"

    # === Predict for next 10-minute interval ===
    next_hour = current_hour
    next_minute = current_minute + 10
    if next_minute >= 60:
        next_hour += 1
        next_minute %= 60

    mask = (df['hour'] == next_hour) & (df['minute'] == next_minute)
    avg_power_next_10min = df.loc[mask, 'total_power'].mean()

    predicted_peak_intensity = get_peak_intensity(row['predicted_total_power'])
    avg_power_peak_intensity = get_peak_intensity(avg_power_next_10min)
    peak_status = is_peak_hour(next_hour, row['season'])

    # === Construct prompt ===
    prompt = f"""
    **Response should not be longer than 10 to 12 lines the format should be in points with numbering with no dashes and any other thing.**
    The total predicted power consumption for the next 10-minute interval ({next_hour:02d}:{next_minute:02d}) is: {row['predicted_total_power']:.2f} kW
    Peak Intensity for Predicted Power: {predicted_peak_intensity} and  temperature: {temp}, Humidity {humidity}, Wind {wind}, GeneralDiffuseFlows {gdiff}, DiffuseFlows {ddiff}, Day {day}, Month {month}
    The average total_power for this time slot across all days is: {avg_power_next_10min:.2f} kW
    Peak Intensity for Average Power: {avg_power_peak_intensity}
    According to the data collected from local authorities the current time is under peak consumption time = {'Yes' if peak_status else 'No'}
    The season is {row['season'].capitalize()} .
    **These line separators are compuslory <br />**
    **Only points not ike these are the recommendation and content should be like for tailwindcss+Html
    
    Peak Times
        'winter': 'months': [12, 1, 2],  'start': 17, 'end': 21,
        'spring': 'months': [3, 4, 5],   'start': 18, 'end': 22,
        'summer': 'months': [6, 7, 8],   'start': 19, 'end': 23,
        'autumn': 'months': [9, 10, 11], 'start': 18, 'end': 22,

    **
    So you can suggest like apply inverter air conditioners, prefer using air cooler if peak time, turn off unnecessary lights,
    and insights which you can give please be broad and brief in your recommendations and **Keep this thing in mind that
    the above discussed recommendations are only for summer season and there can be any
    season amend your response according to the season**.
    Hint:
    1. Tell how electrical appliances specially for that season can be used effectively.
    2. How personal habits can influence power saving.
    3. How that just making minor efforts can save from heavy bills.
    4. Like if it is peak time then you can suggest to turn off the AC but if not peak time then you can keep AC turned on but keep unnecessary appliances off.

    """

    # === Send prompt to Gemini ===
    genai.configure(api_key="AIzaSyB18ouOYf2KUy43YuusIk3SM04PdW6TRj4")
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)

    return response.text



