from tkinter import *

screen = Tk()
screen.title("SMarket Prediction")

# self.lag1 = 0
#         self.lag2 = 0
#         self.lag3 = 0
#         self.lag4 = 0
#         self. lag5 = 0
#         self.vol = 0
#         self.tdy = 0

# Adjust size
screen.geometry("500x500")
# set minimum window size value
screen.minsize(300, 300)
# set maximum window size value
screen.maxsize(500, 500)

Label(screen, text='First Name').grid(row=0)
Label(screen, text='Last Name').grid(row=1)
e1 = Entry(screen)
e2 = Entry(screen)
e1.grid(row=0, column=1)
e2.grid(row=1, column=1)

screen.mainloop()