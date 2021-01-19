# Jane Street market prediction

## 1. Overview

### **1.1 Description**

In a perfectly efficient market, buyers and sellers would have all the agency and information needed to make rational trading decisions. As a result, products would always remain at their “fair values” and never be undervalued or overpriced. However, financial markets are not perfectly efficient in the real world.

Even if a strategy is profitable now, it may not be in the future, and market volatility makes it impossible to predict the profitability of any given trade with certainty. 

글로벌 주식 거래를 보고 이익 최대화 하는 모델 만들기

미래의 시장을 예측하기.

예측 모델 만들기.

### **1.2 Evaluation**

- Utility score

  Each row in the test set represents a trading opportunity for which **you will be predicting an `action` value, 1 to make the trade and 0 to pass on it.** Each trade `j` has an associated `weight` and `resp`, which represents a return.

$$
p_i = \sum_j(weight_{ij} * resp_{ij} * action_{ij}),
$$


$$
t = \frac{\sum p_i }{\sqrt{\sum p_i^2}} * \sqrt{\frac{250}{|i|}},
$$
​				where |i| is the number of unique dates in the test set. The utility is 				then defined as:
$$
u = min(max(t,0), 6)  \sum p_i.
$$

>  https://www.kaggle.com/renataghisloti/understanding-the-utility-score-function

- _Pi_ 

  Each row or trading opportunity can be chosen (action == 1) or not (action == 0). 거래가 일어나면 1, 일어나지않으면 0.

  The variable _pi_ is a indicator for each day _i_, showing how much return we got for that day.

  Since we want to maximize u, we also want to maximize _pi_. To do that, we have to select the least amount of negative _resp_ values as possible (since this is the only negative value in my equation and only value that would make the total sum of p going down) and maximize the positive number of positive _resp_ transactions we select.

  u를 최대화 하려면 결국 _`pi`_를 최대화 해야함.

- _`t`_ 

  **_t_** is **larger** when the return for **each day is better distributed and has lower variation.** It is better to have returns uniformly divided among days than have all of your returns concentrated in just one day. It reminds me a little of a **_L1_** over **_L2_** situation, where the **_L2_** norm penalizes outliers more than **_L1_**.

  

  Basically, we want to select uniformly distributed distributed returns over days, maximizing our return but giving a penalty on choosing too many dates.

  매일 매일 균등하게 분포되어야함. 그래야 return값을 최대화 할 수 있음.

- t is simply the annualized sharpe ratio assuming that there are 250 trading days in a year, an important risk adjusted performance measure in investing. If sharpe ratio is negative, utility is zero. A sharpe ratio higher than 6 is very unlikely, so it is capped at 6. The utility function overall try to maximize the product of sharpe ratio and total return.

  t는 샤프지수.

---

## 2. Implementing.

>https://www.kaggle.com/vivekanandverma/eda-xgboost-hyperparameter-tuning
>
>https://www.kaggle.com/wongguoxuan/eda-pca-xgboost-classifier-for-beginners
>
>https://www.kaggle.com/smilewithme/jane-street-eda-of-day-0-and-feature-importance/edit

### 2.0 Preprocessing



### 2.1. EDA

**Market Basics:** Financial market is a dynamic world where investors, speculators, traders, hedgers understand the market by different strategies and use the opportunities to make profit. They may use fundamental, technical analysis, sentimental analysis,etc. to place their bet. As data is growing, many professionals use data to understand and analyze previous trends and predict the future prices to book profit.

**Competition Description:** The dataset provided contains set of features, **feature_{0...129}**,representing real stock market data. Each row in the dataset represents a trading opportunity, for which we will be predicting an action value: **1** to make the trade and **0** to pass on it. 

Each trade has an associated weight and resp, which together represents a return on the trade. In the training set, **train.csv**, you are provided a **resp** value, as well as several other **resp_{1,2,3,4}** values that represent returns over different time horizons.

In **Test set** we don't have **resp** value, and other **resp_{1,2,3,4}** data, so we have to use only **feature_{0...129}** to make prediction.

Trades with **weight = 0** were intentionally included in the dataset for completeness, although such trades **will not** contribute towards the scoring evaluation. So we will ignore it.

### 2.2 Using XGBoost Algorithm

