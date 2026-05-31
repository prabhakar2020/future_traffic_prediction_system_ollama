# future_traffic_prediction_system

This creates sample city traffic mock data based on reallife conditions like time, weekdays, weather, schools open, college ops, political_event, mall_opening, special_event, rain,peak hours, and events. It then uses a machine learning model to learn patterns and predict traffic for each hour of the next day.

It will predict tomorrows traffic as **Low, Moderate, High, or Very High**, helping to understand how traffic changes during the day in a simple and realistic way.

I chose Random Forest because it works very well for structured/tabular data like this traffic dataset where we have features such as time, weather, events, and weekday patterns.
It handles non-linear relationships easily, meaning it can learn complex traffic patterns without needing heavy tuning.

Random forest also help us on stable, works well even when data has noise, fast to train and gives good accuracy for baseline forecasting.

Other models like Linear Regression are too simple for this kind of complex behavior, and deep learning models like LSTM are more suitable for large scale time series systems but require more large data, tuning, and computation.

```
I am running this script on my Ubutnu
Python 3.12.3
Description:    Ubuntu 24.04.4 LTS
Release:        24.04
Codename:       noble
```

### Here are the steps to get everything running smoothly on your local machine
1. Update WSL Ubuntu and Install Python Pip
sudo apt update && sudo apt upgrade -y
sudo apt install python3-pip python3-venv -y
2. Create a Dedicated Project Directory & Virtual Environment
mkdir /home/prabhakar/linux_data/jupyter_project
cd /home/prabhakar/linux_data/jupyter_project
3. Create a virtual environment (named .venv):
python3 -m venv .venv
4. Activate the virtual environment
source .venv/bin/activate
5. Install Dependencies
pip install -r requirements.txt
