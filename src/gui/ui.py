from tkinter import *
from src.models.machine import MarketKNN

model = MarketKNN()

screen = Tk()
screen.title("SMarket Prediction")
# Adjust size
screen.geometry("400x400")
# set minimum window size value
screen.minsize(150, 150)
# set maximum window size value
screen.maxsize(400, 400)

name_feature = ['lag1', 'lag2', 'lag3', 'lag4', 'lag5', 'vol', 'tdy']
entries = []

for n in range(7):
    Label(screen, text=name_feature[n]).grid(row=n, pady=5)
    entry = Entry(screen)
    entry.grid(row=n, column=1, pady=5)
    entries.append(entry) 
    

# Function to update the predict label
def update_prediction():
    data = {name_feature[i]: entries[i].get() for i in range(len(name_feature))}
    pred = model.predict(float(data['lag1']), 
                         float(data['lag2']), 
                         float(data['lag3']), 
                         float(data['lag4']), 
                         float(data['lag5']), 
                         float(data['vol']), 
                         float(data['tdy']))
    
    predict_result.config(text=f"Predict Result : {pred}")  # Update the label's text
    

button = Button(screen, text="Predict", command=update_prediction)
button.grid(row=len(name_feature), column=0, columnspan=2, pady=10)

predict_result = Label(screen, text="Predict Result :")
predict_result.grid(row=len(name_feature)+1)


screen.mainloop()