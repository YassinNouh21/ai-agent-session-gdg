from my_code import get_weather_data

cities = ['London', 'Paris', 'Tokyo']

# Calling the get_weather_data function
try:
    weather_data = get_weather_data(cities)
    
    # Extracting the required information
    hottest_city = weather_data['hottest_city']
    coldest_city = weather_data['coldest_city']
    average_temperature = weather_data['average_temperature']
    
    # Printing the results
    print(f'Hottest City: {hottest_city}')
    print(f'Coldest City: {coldest_city}')
    print(f'Average Temperature: {average_temperature}°C')
except Exception as e:
    print(f'An error occurred: {e}')