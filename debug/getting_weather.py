
try:
    cities = ['London', 'Paris', 'Tokyo']
    weather_data = get_weather_data_for_cities(cities)

    temperatures = [data['temperature'] for city, data in weather_data.items()]
    max_temp_city = max(weather_data, key=lambda x: weather_data[x]['temperature'])
    min_temp_city = min(weather_data, key=lambda x: weather_data[x]['temperature'])
    avg_temp = sum(temperatures) / len(temperatures)

    print(f"Hottest city: {max_temp_city}, Temperature: {weather_data[max_temp_city]['temperature']}°C")
    print(f"Coldest city: {min_temp_city}, Temperature: {weather_data[min_temp_city]['temperature']}°C")
    print(f"Average temperature: {avg_temp}°C")

except Exception as e:
    print(f'An error occurred: {e}')