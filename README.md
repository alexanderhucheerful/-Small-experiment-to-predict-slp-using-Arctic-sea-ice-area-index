#  Small experiment to predict slp using Arctic sea ice area index
  
  dmi is Arctic sea ice area index 
  python3 train.py --> to predict the future Arctic sea ice area index by history data ,datasets use window method to split the smaples
  python3 test.py  -->  use the above the predict index to predict glob slp value but when i set the the scope is global it has      Underfitting,so mybe set small scope is can do  more accurate prediction
