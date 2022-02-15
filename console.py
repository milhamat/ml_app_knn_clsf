import machine as mcn

lag1 = 0.28
lag2 = -0.60
lag3 = -2.90
lag4 = 2.00
lag5 = 5.79
vol = 1.59
tdy = 2.75

ML = mcn.MarketKNN(lag1, lag2, lag3, lag4, lag5, vol, tdy)

ML.predict()