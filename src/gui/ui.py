import random
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

next = 0
k=0
for n in range(7):
    Label(screen, text=name_feature[n]).grid(row=n, pady=5)
    entry = Entry(screen)        
    entry.grid(row=n, column=1, sticky = W, pady=5)
    entries.append(entry)
    next+=1 
    k+=1
    
    
def gen_random():
    for i, entry in enumerate(entries):
        if name_feature[i] in ['lag1', 'lag2', 'lag3', 'lag4', 'lag5', 'tdy']:
            random_value = round(random.uniform(-5.0, 6.0), 2)
        elif name_feature[i] == 'vol':
            random_value = round(random.uniform(0.4, 3.2), 2)
        entry.delete(0, END)  # Clear the existing value in the Entry widget
        entry.insert(0, str(random_value))  # Insert the random value as a string
    
    

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
button.grid(row=len(name_feature), column=0, columnspan=1, pady=5)

rd_button = Button(screen, text="Random", command=gen_random)
rd_button.grid(row=len(name_feature), column=1, columnspan=1, pady=5)

predict_result = Label(screen, text="Predict Result :")
predict_result.grid(row=len(name_feature)+2)


screen.mainloop()