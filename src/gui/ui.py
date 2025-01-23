from tkinter import *

screen = Tk()
screen.title("SMarket Prediction")
# Adjust size
screen.geometry("500x500")
# set minimum window size value
screen.minsize(300, 300)
# set maximum window size value
screen.maxsize(500, 500)

name_feature = ['lag1', 'lag2', 'lag3', 'lag4', 'lag5', 'vol', 'tdy']
entries = []

for n in range(7):
    print(n)
    Label(screen, text=name_feature[n]).grid(row=n, pady=5)
    
    entry = Entry(screen)
    entry.grid(row=n, column=1, pady=5)
    entries.append(entry) 

button = Button(screen, text="Predict")
button.grid(row=len(name_feature), column=0, columnspan=2, pady=10)

predict = Label(screen, text="Up")
predict.grid(row=len(name_feature)+1)




screen.mainloop()