#  XGBoost?

it is contents of ensemble learning.

>https://www.youtube.com/watch?v=VHky3d_qZ_E
>
>https://www.youtube.com/watch?v=1OEeguDBsLU&list=PL23__qkpFtnPHVbPnOm-9Br6119NMkEDE&index=4
>
>https://www.youtube.com/watch?v=4Jz4_IOgS4c
>
>https://www.youtube.com/watch?v=VkaZXGknN3g&feature=youtu.be
>
>https://bkshin.tistory.com/entry/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-15-Gradient-Boost

## 2.2.1 Ensemble learning

x - dataset, y - error rate, each color is algorithm.

- No Free Lunch Theorem.
  - any classification method cannot be superior or inferior overall

Ensemble means harmony or unity.

When we predict the value of a data, we use one model. But if we learn several models in harmony and use their predictions, we'll get a more accurate estimate.

Ensemble learning is a machine learning technique that combines multiple decision trees to perform better than a single decision tree. The key to ensemble learning is to combine several weak classifiers to create a Strong Classifier. This improves the accuracy of the model.

### 2.2.1.1 Bagging

Bagging is Bootstrap Aggregation. Bagging is a method of aggregating results by taking samples multiple times (Bootstrap) each model.



![bagging](https://blog.kakaocdn.net/dn/b4wG8O/btqyfYW98AS/YZBtUJy3jZLyuik1R0aGNk/img.png)

First, bootstrap from the data. (Restore random sampling) Examine the bootstrap data to learn the model. It aggregates the results of the learned model to obtain the final result value.

Categorical data aggregates results in Voting, and Continuous data is averaged.

When it's categorical data, voting means that the highest number of values predicted by the overall model is chosen as the final prediction. Let's say there are six crystal tree models. If you predicted four as A, and two as B, four models will predict A as the final result by voting.

Aggregating by means literally means that each decision tree model averages the predicted values to determine the predicted values of the final bagging model.

Bagging is a simple yet powerful method. Random Forest is representative model of using bagging.



### 2.2.1.2 Boosting

Boosting is a method of making weak classifiers into strong classifiers using weights. The Bagging predicts results independently of the Deicison Tree1 and Decision Tree2. This is how multiple independent decision trees predict the values, and then aggregate the resulting values to predict the final outcome. 

Boosting, however, takes place between models. When the first model predicts, the data is weighted according to its prediction results, and the weights given affect the next model. Repeat the steps for creating new classification rules by focusing on misclassified data.

![img](https://blog.kakaocdn.net/dn/kCejr/btqyghvqEZB/9o3rKTEsuSIDHEfelYFJlk/img.png)



## 2.2.2 Different of Bagging and Boosting



![img](https://blog.kakaocdn.net/dn/bwr6JW/btqygiHRbRk/cy5hbDAPpTjCG7xa6UWxi0/img.png)



Bagging is learned in parallel, while boosting is learned sequentially. After learning once, weights are given according to the results. Such weights affect the prediction of the results of the following models.

High weights are given for incorrect answers and low weights for correct answers. This allows you to focus more on the wrong answers in order to get them right.

Boosting has fewer errors compared to bagging. That is, performance is good performance. However, the speed is slow and there is a possibility of over fitting. So, which one should you choose between bagging or boosting when you actually use it? It depends on the situation. If the low performance of the individual decision tree is a problem, boosting is a good idea, or over-fitting problem bagging is a good idea.



# 2.2 XGBoost : A scalable Tree Boosting System(eXtreme Gradient Boosting)

https://xgboost.readthedocs.io/en/latest/

https://towardsdatascience.com/https-medium-com-vishalmorde-xgboost-algorithm-long-she-may-rein-edd9f99be63d

Optimized Gradient Boosting algorithm through parallel processing, tree-pruning, handling missing values and regularization to avoid over-fitting/bias

![그림1](D:\노트북\hdd\공부\인턴\그림1.png)



**XGBoost = GBM + Regularization**



Red part is Regularization term.

T : weak learner(node)

w : node score



Therefore, it can be seen that the regularization term of XGboost prevents overfitting by giving penalty to loss as the tree complexity increases.

## 2.2.1 Split finding Algorithm

- Basing exact greedy algorithm
  - Pros: **Always find the optimal split** point because it enumerates over *all possible* splitting points greedily
  - Cons: 
    - Impossible to efficiently do so when the data does not fit entirely into memory
    - Cannot be done under a distributed setting

- Approximate algorithm
  - Example
    - Assume that the value is sorted in an ascending order
    - Divide the dataset into 10 buckets
    - Global variant(Per tree) vs Local variant(per split)

## 2.2.2 Sparsity-Aware Split Finding

- In many real-world Problems, it is quite common for the input x to be sparse
  - presence of missing values in the data
  - frequent zero entries in the statistics
  - artifacts of feature engineering such as one-hot encoding
- Solution : Set the default direction that is learned from the data

## 2.2.3 System Design for Efficient Computing

- The most time-consuming part of tree learning
  - to get the data into sorted order
- XGBoost propose to store the data in in-memory units called block
  - Data in each block is stored in the compressed column(CSC) format, with each column sorted by the corresponding feature value
  - This input data layout only needs to be computed once before training and can be reused in later iterations.
- Cache-aware access
  - For the exact greedy algorithm, we can alleviate the problem by a cache-aware prefetching algorithm
  - For approximate algorithms, we solve the problem by choosing a correct block size
- Out-of-core computing
  - Besides processors and memory, it is important to utilize disk space to handle data that does not fit into main memory
  - To enable out-of-core computation, the data is divided into multiple blocks and store each block on disk
  - To enable out-of-core computation, we divide the data into multiple blocks and store each block on disk
  - It is important to reduce the overhead and increase the throughout of disk IO
- Block Compression
  - The block is compressed by columns and decompressed on the fly by an independent thread when loading into main memory
  - This helps to trade some of the computation in decompression with the disk reading cost
- Block Sharding
  - A pre-fetcher thread is assigned to each disk and fetches the data into an in-memory buffer
  - The training thread then alternatively reads the data from each buffer.
  - This helps to increase the throughput of disk reading when multiple disks are available.

> https://www.youtube.com/watch?v=VkaZXGknN3g&feature=youtu.be
>
> http://machinelearningkorea.com/2019/09/28/xgboost-%EB%85%BC%EB%AC%B8%EB%94%B0%EB%9D%BC%EA%B0%80%EA%B8%B0/
>
> https://soobarkbar.tistory.com/32
>
> http://dmlc.cs.washington.edu/data/pdf/XGBoostArxiv.pdf
>
> https://wiki.math.uwaterloo.ca/statwiki/index.php?title=summary
>
> XGBoost Paper Review
>
> 1. Introduction
>
>    - Gradient Boosting + Extreme
>    - 왜 유명한가? Scalability(확장성)
>    - Innovations
>      - Tree Boosting 개선
>      - Split Finding Algorithms 제시
>    - Systematic
>      - System Design
>
> 2. Tree Boosting in a Nutshell
>
>    2.1 Regularized Learning Objective
>
>    - 기본적인 트리 부스팅
>
>      EX) 
>
>      	- AdaBoost :첫 트리의 결과를 본다. 덜 잘못 예측된 값에 weight를 주고 f1 f2 f3 에 대해서............
>      	- GBM : f1에서 예측하고, label과 비교했을 떄 residual 이 나타나면 그 residual를 예측함으로써 부족한부분을 메꿔바는 방법으로 진행
>
>     - XGboost Base Learner : CART(Classification And Regression Trees)
>
>       - How can we get w, weight of the node? How can we evaluate q, quality of CART?
>
>         -> Loss Function으로 구함.
>
>    2.2 Gradient Tree Boosting
>
>    ![img](https://blog.kakaocdn.net/dn/bhQVf5/btqBCTRuFDf/MkTBEw40tvzVCg1sbsNz7k/img.png)
>
>    ​		앞에는 Training Loss, 뒤에는 Regularization: Complexity of the Trees.
>
>    ![img](https://blog.kakaocdn.net/dn/Gb39Z/btqBBUi6Qub/rGenMsRirkThImtmgY07gK/img.png)
>
>    - 본 Loss Function은
>
>      - Function을 parameter로 사용하기 때문에,
>
>      - Numerical Vector가 아니기에
>
>        -> Euclidean space를 활용하여 optimization이 불가능하다.
>
>      -> Additive Training (Boosting)
>      $$
>      \hat{y}^t = \sum_{k=1}^tf_k(x_i) =\hat{y}^{t-1} + f_t(x_i) ..... (1)
>      $$
>      t 번째 Prediction = t-1번째 Prediction + Added Function
>
>      loss function 에 대입하면.
>      $$
>      L(\phi) = \sum_{k=1}^tl(y_i,\hat{y}^{t-1} + f_t(x_i)) + \Omega(f_t)
>      $$
>      Taylor Expansion 진행
>
>      - Taylor Series(Taylor Expansion)
>
>        - 미지의 함수 f(x)를 다항함수로 근사.
>
>          - 이해 불가한 함수를 이해의 영ㅇ역인 다항함수로 표현
>
>          - 깔끔한 대신 '근사'
>            $$
>            \mathcal{L}^{(t)}\approx \sum_{i=1}^n\ell(y_i,\hat{y}_i^{(t-1)})+g_if_t(\mathbf{x}_i)+\frac{1}{2}h_if_t^2(\mathbf{x}_i)+\Omega(f_t),
>            $$
>
>            $$
>            \ell(y_i,\hat{y}_i^{(t-1)}) 를 F(X), 로 생각하면 이때, f_t(x_t)의 값은 이전 값들의 합보다는 비교적 작을 것이다.
>            $$

> ![Image for post](https://miro.medium.com/max/520/1*jdYI5gjPUXSoyf-T4YrLHA.png)
>
> ![Image for post](https://miro.medium.com/max/449/1*KyybUoc-tCT6466FkWp3Cw.png)

>    -  Why choose T over n?
>       - Tree의 관점으로 Loss를 봐야하니 T로 규합	n = instances, T= Leaf의 갯수

![Image for post](https://miro.medium.com/max/380/1*ce2L033-2HfZHC-i7BCgRg.png)

> ![Image for post](https://miro.medium.com/max/579/1*QL-uJ9zBKrT19ugCYrbO4A.png)

> gt부분 sum(ㅜ개의 데이터의 각기의 1차미분 * 트리 결과값) = 아랫부분 gmxi부분모든 leaf(T)에 대한 (Leaf j의 instance들의 1차 미분 합) * (Leaf Weight j) 의 합



![Image for post](https://miro.medium.com/max/379/1*RE57Lvsbure_KICFBifvtA.png)

> ![Image for post](https://miro.medium.com/max/296/1*qGF58wI3LzNYEzU9U2VEdw.png)
>
> Tree q(x)가 고정일 때, Leaf의 Optimal Weight, 는

![Image for post](https://miro.medium.com/max/329/1*pgAVG1tCOZrDbO5_3NDn0Q.png)

![Image for post](https://miro.medium.com/max/569/1*J_VT3VcoKLL-yFRV9eBgAg.png)

> Gain 은 Split 이후 Loss Reduction을 나타내며, 분할을 했을 때 감소폭이 큰 분할을 선택한다.
>
> 2.3 Shrinkage and Column Subsampling
>
> - Shrinkage
>
>   새롭게 추가되는 Tree에 를 곱해, 각 트리의 영향을 낮춘다. Stochastic learning의 Learning Rate. -> 미래의 추가되는 tree, 너만 믿는다.
>
> - Column Subsampling
>
>   Random Forest에서 사용 되어온 기법.
>
> 3. Split Finding Algorithms.
>
>    3.1 Basic Exact Greedy Algorithm![Image for post](https://miro.medium.com/max/654/1*HwpFPnJ1_Oi4fDDsLuuEvA.png)
>
>    - 모든 경우의 수를 다 탐색하자.
>
>      feature 에 대한 split을 찾는 과정
>
>      전부다 iterate하는 것은 비용이 너무 크기 때문에, 효과적으로 진행하기 위해선 **Soring**을 해야한다.
>
>    - 장점 : 최적해 보장, 단점: 시간 --->> Approximate Algorithm(최적해 근사, 시간 감소)
>
>    3.2 Approximate Algorithm![img](http://machinelearningkorea.com/wp-content/uploads/2019/09/Screenshot-from-2019-09-28-19-13-29.png)
>
>    - Feature분포의 Percentile을 통해 Candidate Splitting Points를 구성
>    - (Bucket) Split으로 Feature들을 나눈 뒤, 통계량을 구한 뒤, 최고의 Solution.
>    - Parallelization이 가능해진다!
>    - Global Variant vs Local Variant
>      - Global Variant : Proposes all the candidate splits during the initial phase
>      - Local Variant: Re-propose after splits, Refinement for each split(bigger tree에 어울림)
>    - Approximation factor.->Percentile을 얼마나 잘게 나눌지에 대한 파라미터.
>
>    3.3 Weighted Quantile Sketch
>
>    - Sketch Algorithm
>
>      Sample Data로 Sketch를 하여 Original Data Distribution 파악
>
>    - Quantile Sketch Algorithm
>
>      각 Quantile의 Sketch로 Original Data Distributionㄹ을 파악
>
>    - Weighted Quantile Sketch Algorithm
>
>      Normal Quantile: 각 quantile의 데이터 개수가 같다..
>
>      Weighted quantile: 각 quantile의 sum of weights가 같다.![img](http://machinelearningkorea.com/wp-content/uploads/2019/09/Screenshot-from-2019-09-28-19-22-44.png)
>
>      hi가 Weight로 작동.
>
>    3.4 Sparsity-Aware Split Finding
>
>    - Dense Data를 사용하면 좋지만 현실에는 Sparse Data가 가득하다.
>    - XGBoost가 좋은점은 Scalability를 통해서 이런 케이스도 아우를수 있다.
>    - 많은 데이터들의 input variable은 sparse한 경우가 많다.
>    - Sparsity 의 대표적인 원인
>      - Presence of missing values in the data
>      - Frequent zero values
>      - Artifacts of feature engineering such as One-Hot Encoding
>    - Sparsity Aware! > Add a Default Direction in each tree node, 모든 결측치는 기본 방향으로 이동한다!.
>    - 갈 수 있는 방향은 두가지이며, 데이터로부터 학습!
>    - ![Algorithm 3.png](https://wiki.math.uwaterloo.ca/statwiki/images/thumb/2/28/Algorithm_3.png/500px-Algorithm_3.png)
>    - input(i) : 현재 노드의 instance set
>    - input(i_k) : k feature의 non-missing data
>    - input(d) = feature dimension
>    - gain = 0, G H 각 x의 1차미분합 2차미분합
>    - enumer~~ 먼저 모든 missing value를 오른쪽 노드로 이동
>    - for i in~~ 오름차순으로 정리한 xjk 에 대하여 Score를 계산하고 max를 통해 update(feature k, instance j)
>    - enumer~~ 먼저 모든 missing value를 왼쪽
>    - for j in ~~ 오름차순으로 정리한 xjk 에 대하여 max를 통해 update(feature k, instance j)
>
> 4. System Design
>
>    innovation이 알고리즘파트와 시스템파트로 이루어짐. 알고리즘 파트는 시스템파트가 가능할 수 있도록, 알고리즘을 그에 맞춰서 변경 해준것에 불가하다 는 말이 무색할 수 있다. 
>
>    4.1 Column block For parallel Learning
>
>    - 데이터 저장 방법 : Row vs Column Orientation
>    - 보통 Row operation -> Transactional Processing, All the columns are required : 하나의 Column만 비교하고 싶을 때도 모든 Column을 함께 사용해야 한다.
>    - Column Orientation : Only relevant columns are required, Reports are aggregates
>    - Sorting은 Time을 많이 사용함
>    - Data를 in-memory unit인 Block 에 담아서 해결하자.
>    - 각 Block의 데이터는 Compressed column(CSC) format으로 저장. 훈련 전에 미리 Compute해두고 지속적으로 사용. 
>
>    4.2 Cache-Aware Access
>
>    - I/O Speed: CPU Cache > Memory > Disk
>    - 즉, Cache로만 Gradient Statistics 계산 및 활용을 할 수 있다면 속도 발전에 도움
>
>    4.3 Blocks for Out-of-core computation
>
>    - Machine을 최고의 효율로 사용하자
>    - 앞에서는 Cache와 Memory의 관점만 얘기했지만 Disk도 중요.
>    - Disk Reading에 많은 시간이 걸려 Overhead 발생
>
> 5. Related Works
>    - XGBoost
>      - Gradient Boosting : Additive Optimization in functional Space
>      - Regularization : Prevent Over fitting
>      - Column Sampling : Random Forest