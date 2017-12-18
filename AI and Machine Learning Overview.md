## Goals and Purpose

- Derive discoveries, information, patterns, trends, and actionable insights
    + Automated, ad-hoc, and self-service
- Transform data into value
- Build data products providing actionable information while abstracting away technical details
- Improving products for user benefit and experience
- Driving business decisions and solutions
    + Improving the decisions your business makes
        * Decision science uses data to analyze business metrics — such as growth, engagement, profitability drivers, and user feedback — to inform strategy and key business decisions.
        * Decision science uses data analysis and visualization to inform business and product decisions.
    + Inform strategic decisions
    + Inform product changes and drive company KPIs
    + Shift from HiPPO decision making to data-driven decision making
- Automated decision making, predictions, recommendations, and deeper insights
- Competitive advantage, differentiation, future-proofing, and opportunity costs
- Complement business intelligence functions
- Predict and advise
- Shifting between deductive (hypothesis-based) and inductive (pattern- based) reasoning (ref. Booz, Allen, Hamilton)
    + Deductive
        * Formulate hypotheses about relationships and underlying models
        * Carry out experiments with the data to test hypotheses and models
    + Inductive
        * Exploratory data analysis to discover or refine hypotheses
        * Discover new relationships, insights and analytic paths from the data
- Business top-level goals
    + Increase ROI and ROA
    + Increase revenue
    + Increase profit
    + Decrease costs
    + Predict and/or reduce risk
    + Increase operational efficiency
    + Reduce waste
    + Increase customer acquisition, retention, and growth
    + Maximize customer retention and minimize churn/loss (particularly to competitors)
    + Improve customer service
    + Enhance business development
    + Improve business governance
    + Improve business processes
    + Drive data and evidence-driven decisions
    + Improve business agility
    + Decision management and inform decisions
    + Drive new business models
    + Discover new products and services
    + Business reporting and analysis
- Customer goals
    + Great user experience and design
        * Self-evident, or at least self-explanatory
        * Easy to use
        * Easy to understand
        * No extra fluff or unnecessary information
    + Products that are sticky, i.e., maximum stickyness
    + Highly personalized experiences
- Goals for artificial intelligence in the future
    + It just works
    + Great user experience
    + Helps humanity, not hurts
- Generating and delivering actionable insights via
    + Story telling
    + Writing
    + Speaking
    + Reports
    + Dashboards
    + Visualizations
    + Presentation
    + Asynchronous messaging, notifications, insights, and alerts

## Types

- AI (encompasses machine learning and other fields)
    + Weak/Narrow - Non-sentient artificial intelligence that is focused on one narrow task
    + Deep - Machine learning based on deep neural networks, i.e., those with more than one hidden layer, each which learns components of the bigger picture
        * Multiple hidden layers allow deep neural networks to learn simple features of the data in a hierarchy, which are combined by the network
    + Shallow - One so-called hidden layer
        * Can be very wide to achieve similar results to deep, but not as efficient and requires more neurons
    + Soft - Not really used
    + Applied - Building ‘smart’ systems or enhancing or extending software applications, where each AI is built for a single purpose
    + AGI, Strong, Full, Hard, Complete
        * Hard/Complete (AI-hard/AI-complete) - The most difficult problems, where the difficulty of these computational problems is equivalent to that of solving the central artificial intelligence problem—making computers as intelligent as people, or strong AI
        * Full - Synonymous with strong and AGI
        * Strong - Strong AI's goal is to develop artificial intelligence to the point where the machine's intellectual capability is functionally equal to a human's.
        * AGI - A machine with the ability to apply intelligence to any problem, rather than just one specific problem
            - The intelligence of a machine that could successfully perform any intellectual task that a human being can
- Machine learning (ML, and subset of AI)
    + Primary
        * Supervised
            - Regression
            - Classification
        * Unsupervised
            - Clustering
            - Anomaly detection
        * Semi-supervised
        * Reinforcement learning
    + Other
        * Transfer learning
        * Recommender systems
            - Content-based and collaborative-based filtering
        * Ensemble methods
            - Bagging (random forests) and boosting

## AI Related Terms

- Sentient - Able to perceive or feel things
- Consciousness - The state of being awake and aware of one's surroundings (opposite is dead or inanimate object)
    + The awareness or perception of something by a person
    + The fact of awareness by the mind of itself and the world
- Mind - The element, part, substance, or process that reasons, thinks, feels, wills, perceives, judges, etc.
- Think - To employ one’s mind rationally and objectively in evaluating or dealing with a given situation
- Artificial (aka machine or synthetic) consciousness - The aim of the theory of artificial consciousness is to "Define that which would have to be synthesized were consciousness to be found in an engineered artifact"
- Sensing
- Learning
- Remembering
- Reasoning
- Transcendent 
- Conscious 
- Self-aware
- Latent Intelligence


## AI and ML Process (High-level)

- Ask the right question
- Obtain the data
- Parse and process the data
- EDA - statistical analysis and data visualization
- Choose models/algorithms and performance metric
- Iterate and improve performance
- Deliver/communicate/present results

## AI and ML Tradeoffs, Considerations, and Constraints (High-level)

**Model Selection**
+ No Free Lunch (NFL) theorem
    * There is no model that is a priori guaranteed to work better
    * Must evaluate multiple and make reasonable assumptions
+ Parametric vs non-parametric
    * Parametric examples
        - Simple and multiple linear regresion
    * Non-parametric examples
        - Decision trees
        - KNN - k nearest neighbors
        - SVMs
+ Generative vs discriminative models<sup>1</sup>
    * Generative
        - Gaussian mixture model
        - Hidden Markov model
        - Probabilistic context-free grammar
        - Naive Bayes
        - Averaged one-dependence estimators
        - Latent Dirichlet allocation
        - Restricted Boltzmann machine
        - Generative adversarial networks
    * Discriminative
        - Linear regression
        - Logistic regression
        - Support Vector Machines
        - Maximum Entropy Markov Model
        - Boosting (meta-algorithm)
        - Conditional Random Fields
        - Neural Networks
        - Random forests
+ Model complexity
+ Model assumptions (e.g., linearity)
+ Handling of redundant or irrelevant features
+ Ability to perform with small data sets
+ Complexity vs parsimony
+ Ability to automatically learn feature interactions
+ Classification via definitive vs probabilistic assigment
+ Output types
    * Continuous numeric
    * Probability: 0 to 1
    * Classification
        - Binary
        - Multiclass/ multinomial
            + One vs all or one vs rest (probability per class and ranked)
            + One vs one (binary classifier for every class)
        - Multilabel (assign to multiple classes)
            + Multiple binary labels
        - Multioutput
            + Multiple labels, each label can be multiclass (more than two classes)
+ Kernel selection (e.g., SVM)
    * Linear
    * Polynomial
    * Gaussian
    * RBF
+ Feature engineering tractability
    * Use NN and DL if intractable
+ Neural networks and deep learning-specific
    * Artificial neurons
        - Linear threshold unit (LTU)
    * Inputs
        - Sound/audio
        - Time series
        - Text
        - Image
        - Video
        - Unlabeled and/or unstructured data
        - Labeled data
            + Can be difficult, time consuming, and/or costly to obtain
            + Consider a service, e.g., Mechanical Turk, CrowdFlower, ...
    * Number of hidden layers
    * Layer type
        - Fully connected (aka dense)
        - Restricted Boltzman Machine (RBM)
        - Autoencoders
    * Number of neurons per layer
    * Network topology and interconnections and interconnection types between neurons
    * Network depth vs width
        - Parameter efficiency (increased depth reduces neurons required)
    * Outputs and output type
        - Output neurons
            + Single neuron: real-valued numeric (regression)
            + Single neuron: binary classification (classification)
            + Multiple neurons: class scores/probabilities (classification)
            + Multiple neurons: binary assignment per class (classification)
        - Output controlled by output layer activation functions
    * Activation functions and forward propogation
        - Output values, e.g., 0 to 1, -1 to 1, ...
        - Nonzero is 'activated'
        - Examples
            + Linear
            + Rectified linear units (ReLU), Leaky ReLU, Randomized leaky ReLU (RReLU), Parametric leaky ReLU (PReLU)
            + Sigmoid and hard sigmoid
            + Tanh, hard Tanh, rational Tanh
            + Softmax and hierarchical softmax
            + Softplus
            + Softsign
            + Exponential linear unit (ELU)
                * Slower, but very high performing
            + Cube
    * Architecture type <sup>4</sup>
        - Unsupervised Pretrained Networks (UPNs)
            + Autoencoders
            + Deep Belief Networks (DBNs)
            + Generative Adversarial Networks (GANs)
        - Convolutional Neural Networks (CNNs)
            + Image modeling
        - Recurrent Neural Networks
            + Sequence modeling
                * E.g., Long Short-Term Memory (LSTM)
        - Recursive Neural Networks
    * Algorithms
        - First and second order
            + First-order partial derivatives (Jacobians) vs Second-order partial derivatives (the Hessians)
    * Translation invariance

**Model Performance**
- Overfitting and quality of fit (aka bias vs variance)
    + Noise - errors and outliers
    + Not enough data
    + Overly complex model
    + Solutions
        * More data
        * Reduce noise and/or increase the signal-to-noise ratio
        * Reduce model complexity (see Model Complexity and Reduction section)
- Underfitting
    + Opposite of overfitting
    + Solutions
        * Increase model complexity and reduce regularization (if applicable)
        * Feature engineering
        * Select more powerful and complex model (e.g., neural networks)
- Performance metric selection
    + Precision vs recall
    + ROC and AUC
- Performance metric tradeoffs
    + Cost of false positives vs false negatives
- Error types
    + Error types 1 and 2
    + Out-of-sample error (generalization error)
        * Sum of: bias error + variance error + irreducible error
    + In-sample vs out-of-sample errors
- Inability to achieve desired performance level
    + Ill-posed problem
    + Intractability
        * Bad data and/or bad algorithm
- Collinearity, multicollinearity, correlation, ...
- Confounding variables
- Missing features
- Global vs local minima
- Ensemble learning
- Linear separability
- Choice of loss function
    + Regression examples
        * Mean squared error (MSE)
        * Mean squared log error (MSLE)
        * Root mean squared error (RMSE)
        * Mean absolute percentage error (MAPE)
        * Mean absolute error (MAE)
    + Classification examples
        * Hinge loss
        * Logistic loss
        * Negative log likelihood
    + Reconstruction
        * Entropy loss

**Model Complexity and Reduction**
- Subset selection
    + Best subset selection
    + Stepwise selection (forward and backward)
- Shrinkage and regularization
    + Ridge regression
    + The Lasso
    + Elastic Net
    + Neural networks
        * Early stopping
        * L1 and L2 regularization
        * Dropout
        * Max-norm regularization
        * Data augmentation
- Dimension reduction
    + PCA
    + Partial least squares
- Tree methods (e.g., pruning, ...)
- Feature selection, engineering, and extraction
    + Collinearity, multicollinearity, correlation, ...
    + Confounding variables
    + Missing features
    + Feature extraction via dimensionality reduction

**Model Training and Learning**
- Learning type
    + Supervised
    + Unsupervised
    + Semi-supervised
        * Much more unlabled data
        * examples
            * Apply a label to clusters
            * Fine tuning neural networks trained with unsupervised methods
    + Reinforcement
        * Agent
        * Environment
        * Actions
        * Policy
        * Rewards
    + Transfer
        * Reuse similar network lower layers
        * Requires much less training data
        * Speeds up training
        * Frozen layers and stop gradient
        * Model zoo for models
- Machine learning algorithm families <sup>5</sup>
    + Information-based learning
    + Similarity-based learning
    + Probability-based learning
    + Error-based learning
- Offline (batch) vs online learning
    + Offline
        * Train and deploy
    + Online (incremental)
        * Limited computing resources
        * Usually done offline
        * Requires close monitoring of input data for data quality and anomaly/outlier detection, and also to model performance over time
    + Out-of-core (type of online learning)
        * Train on datasets too big to fit into system memory
        * Incremental loading of data into memory and training
- Instance-based vs model-based learning
    + Comparison vs pattern/relationship detection
- Iterative learning
    + Gradient descent
        * Batch (full dataset)
        * Mini-batch (small datasets)
        * Stochastic gradient descent (single instance)
    + KNN
- Gradient descent
    + Random initialization
    + Convergence vs divergence
    + Global vs local minimum
    + Convex vs non-convex functions
    + Weight space vs error/loss space
    + Early stopping
- Gradient descent vs normal equations
    + Depends on computational complexity and speed/time
- Cost or loss function selection
    + MSE
    + Exponential log likelihood
    + Cross entropy
    + Multiclass Cross Entropy
    + RMSE Cross Entropy
    + Squared Loss
    + Negative Log Likelihood
- Vanishing gradients, exploding gradients, batch normalization (BN), and gradient clipping
- Unstable gradients
- Activation function saturation
- Dying or comatose ReLUs
    + Only output 0, neuron goes dead, once dead -> stays dead
- Internal Covariate Shift problem
- Unsupervised Pretraining
    + RBM or autoencoders
- Model pretraining, transfer learning, and frozen layers
    + Pretraining for weight initialization
    + Model zoos
    + Lower layer reuse
    + 
- Max margin learning
- Initilization strategy
    + He initialization
    + Glorot initialization
    + Xavier initialization
    + Random initializations
    + ESN-based Initialization
- Speed up DNN training <sup>5</sup>
    + Apply a good initialization strategy for the connection weights
    + Use a good activation function
    + Use Batch Normalization
    + Use parts of a pretrained network
    + Use a faster optimizer (updater) than the regular Gradient Descent optimizer
        * Momentum optimization
        * Nesterov Accelerated Gradient
        * AdaGrad
        * RMSProp
        * Adam
        * AdaADelta
        * SGD
        * Conjugate Gradient
        * Hessian Free
        * LBFGS
        * Line Gradient Descent
- Sparse data
    + Dual averaging, aka Follow The Regularized Leader (FTRL)

**Model Validation, Tuning, and Optimization**
- Resampling
    + Cross-validation
    + Bootstrap
- Hyperparameters
    + Example categories <sup>4</sup>
        * Layer size
        * Magnitude (momentum, learning rate)
        * Regularization (dropout, drop connect, L1, L2)
        * Activations (and activation function families)
        * Weight initialization strategy
        * Loss functions
        * Settings for epochs during training (mini-batch size)
        * Normalization scheme for input data (vectorization)
- Hyperparameter tuning and optimization
    + Grid search
    + Randomized search for large search space
    + Learning rate reduction (learning schedules)
        * Predetermined piecewise constant learning rate
        * Performance scheduling
        * Exponential scheduling
        * Power scheduling
- Ensemble methods
- Bagging and boosting
- Kernel selection (e.g., SVM)
- Learning curves

**Data, Data Sources, and Data Preparation**
- Analytics base table (ABT) and data quality report <sup>5</sup>
- Balanced vs imbalanced data
    + Equal proportion of target values
- Data availability, amount, and depth
    + Small data sets
        * Selecting models that excel with small data sets
        * Sampling noise
    + Moderate to large data sets
        * Sampling bias
    + Sparse data
    + Resources
        * http://static.googleusercontent.com/media/research.google.com/fr//pubs/archive/35179.pdf
        * https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/acl2001.pdf
- Curse of dimensionality and representative data
    + Exponential increase in data needed to prevent loss of predictive power
    + Data must fill as much as the feature space as possible, and be as balanced as possible
- Data quality
    + Signal to noise ratio
    + Data veracity
    + Errors
    + Outliers
    + Missing data
    + NAs
    + Irrelevant features
- Data types
    + Structured data
    + Unstructured data
    + Semi-structured data
    + Streaming data
    + Batch data
- Data acquisition, ingestion, and processing
    + Disparate data sources
- Data preparation
    + Feature scaling
        * Standardization
        * Min-max (normalization)
    + Feature encoding
    + Missing values handling
    + Imputation

**Computing and Infrastructure Requirements and Performance**
- CPU processing speed and computing time
    + Data processing, model training, and prediction speed
- CPU vs GPU
- System memory (RAM)
- Disk storage
- I/O: disk, network, etc.
- Cost
- Training and prediction speed
- Computational complexity

**Real-World AI and Machine Learning**
- AI and machine learnng in production
    + Deploying to production and maintenance
    + Scalability
    + Online vs offline learning
    + Distributed learning
    + Model types and implementation lanaguages
    + Data access (if required), particularly for disparate data sources
    + Temporal drift, i.e., data changing over time
        * Avoid stale models
    + Model monitoring and accuracy tracking over time
- Working with large datasets <sup>1</sup><sup>,</sup><sup>2</sup>
    + Partition data vs partition execution
    + Software-specific memory allocation
    + Full dataset vs subset (down sampling, e.g., 1/10 to 1/100)
    + Local development vs remote compute resources (e.g., AWS) with lots of RAM
    + Data format transformation
    + Streaming machine learning
    + Progressively loading
    + Direct database connection for machine learning
    + Big data platformes (e.g., Spark and MLLib)
    + Parallel and distributed computing and associated communications cost and complexity
    + Parameter server

**General AI and Machine Learning**
- Model performance vs interpretability and explainability
    + Black box vs non-black box algorithms
- Model complexity vs simplicity (i.e., parsimony)
- Generalization vs representation
- AI limitations
    + Unsupervised learning to some extent
    + Solving multiple problems at once
- Statistical, algorithmic, and cognitive biases
    + Sample or selection bias
    + Confirmation bias
    + Algorithmic bias <sup>3</sup>
- Ethical considerations
- AI and machine learning process
    + CRISP-DM, etc.
- AI and machine learning automations and future advancements
    + Automated learning (AutoML)
        * Auto-sklearn, TPOT, ...
- Execution models
    + Sequencing
        * Serial execution
        * Parallel execution
    + Scheduling
        * Streaming execution
        * Batch execution

**Statistics**
- Prediction vs inference
    + Prediction
        * Model an output variable Y (aka response or dependent variable) as a function of a set of input variables X (aka inputs, features, predictors, independent variables)
            - Estimate f such that: Y = f(X) + err
        * Consider reducible vs irreducible errors
    + Inference
        * Understand how Y varies with X and any underlying relationships between Y and X, particularly wrt. individual predictors and each of their impact on the response
        * Understand the degree to which Y varies with each predictor, e.g., linear, non-linear, ...

## AI Current and Future

- Hype cycle
    + AI winter and trough of disillutionment
- Expectation management
- Actual AI vs simpler machine learning algorithms
- Machine learning by itself is not AI, intelligent systems are built from machine learning models and much more

## AI and Machine Learning Costs

- Financial
- Non-financial

## References

1. [7 Ways to Handle Large Data Files for Machine Learning](https://machinelearningmastery.com/large-data-files-machine-learning/)
2. [Beyond Distributed ML Algorithms: Techniques for Learning from Large Data Sets](https://iwringer.wordpress.com/2015/10/06/techniques-for-learning-from-large-amounts-of-data/)
3. [Joy Buolamwini - TED Talk](https://www.ted.com/talks/joy_buolamwini_how_i_m_fighting_bias_in_algorithms)
4. [Deep Learning by Josh Patterson and Adam Gibson - O'Reilly](https://www.amazon.com/Deep-Learning-Practitioners-Josh-Patterson-ebook/dp/B074D5YF1D/ref=mt_kindle?_encoding=UTF8&me=)
5. [Fundamentals of Machine Learning for Predictive Data Analytics](https://www.amazon.com/Fundamentals-Machine-Learning-Predictive-Analytics-ebook/dp/B013FHC8CM/ref=sr_1_1)
