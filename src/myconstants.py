import pandas as pd

FIGSIZE = (16, 5)
HEADER = 'Power (kW)'
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
days_of_the_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
bank_holidays = pd.to_datetime(["27/03/2024", "28/03/2024", "29/03/2024", "30/03/2024", "31/03/2024", "01/04/2024", "02/04/2024", "03/04/2024", 
                 "06/05/2024", "27/05/2024", "26/08/2024"], format='%d/%m/%Y', utc=True)
spring_term = pd.date_range(pd.to_datetime("2024-01-07", utc=True), pd.to_datetime("2024-03-24", utc=True), freq=pd.Timedelta(days=1))
summer_term = pd.date_range(pd.to_datetime("2024-04-29", utc=True), pd.to_datetime("2024-06-30", utc=True), freq=pd.Timedelta(days=1))
term_time = spring_term.append(summer_term)