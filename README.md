# future_traffic_prediction_system

This creates sample city traffic mock data based on reallife conditions like time, weekdays, weather, schools open, college ops, political_event, mall_opening, special_event, rain,peak hours, and events. It then uses a machine learning model to learn patterns and predict traffic for each hour of the next day.

It will predict tomorrows traffic as **Low, Moderate, High, or Very High**, helping to understand how traffic changes during the day in a simple and realistic way.

I chose Random Forest because it works very well for structured/tabular data like this traffic dataset where we have features such as time, weather, events, and weekday patterns.
It handles non-linear relationships easily, meaning it can learn complex traffic patterns without needing heavy tuning.

Random forest also help us on stable, works well even when data has noise, fast to train and gives good accuracy for baseline forecasting.

Other models like Linear Regression are too simple for this kind of complex behavior, and deep learning models like LSTM are more suitable for large scale time series systems but require more large data, tuning, and computation.
