# Spred_NLP
Natural Language Processing - CSE-538 - Course Project

Contributors:
Anurag Dutt and Srinidhi Bhat 

Aim:
Main aim of this project is to determine the movement of the stocks of the top 500 S&P companies listed in the US Stock Market 
by using Analyst reports called 8-k filings. 

The project ensures to predict the sentiment using no "numerical metric" and only sentences. 

Techniques used:
1) A Bi-LSTM module was developed and used for dependency parsing. The features extracted was fed to seperate CNN,RNN,MLP and CNN-RNN modules. 
2) In our experiment we found CNN-RNN to perform the best.
