
import requests
import time

def get_weather_data(cities, api_key):
    base_url = 'http://api.openweathermap.org/data/2.5/weather'
    results = {}
    total_temp = 0
    hottest_city = ('', -273.15)  # Absolute zero in Celsius
    coldest_city = ('', 100)  # Boiling point of water in Celsius
    
    for city in cities:
        params = {'q': city, 'appid': api_key, 'units': 'metric'}
        response = requests.get(base_url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            temp = data['main']['temp']
            total_temp += temp
            
            if temp > hottest_city[1]:
                hottest_city = (city, temp)
            
            if temp < coldest_city[1]:
                coldest_city = (city, temp)
        else:
            results[city] = 'Error: City not found'
        
        time.sleep(1)  # Rate limiting to avoid API throttling
    
    avg_temp = total_temp / len(cities)
    results['average_temperature'] = avg_temp
    results['hottest_city'] = {'city': hottest_city[0], 'temperature': hottest_city[1]}
    results['coldest_city'] = {'city': coldest_city[0], 'temperature': coldest_city[1]}
    
    return results
