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
    + Products that are sticky
    + Personalized experiences
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
        * Can be very wide to achieve similar results to deep, but not as good or efficient
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
+ Parametric vs non-parametric
    * Parametric examples
        - Simple and multiple linear regresion
    * Non-parametric examples
        - Decision trees
        - KNN - k nearest neighbors
        - SVMs
+ Generative vs discriminative models
+ Handling of redundant or irrelevant features
+ Ability to perform with small data sets
+ Complexity vs parsimony
+ Ability to automatically learn feature interactions
+ Classification via definitive vs probabilistic assigment
+ Kernel selection (e.g., SVM)
+ Neural networks and deep learning-specific
    * Inputs
    * Hidden layers
    * Outputs and output type
        - Output types
            + Real-valued numeric (regression)
            + Probabilities (binary and multiclass classification)
        - Output controlled by output layer activation functions
    * Activation functions and forward propogation
        - Output values, e.g., 0 to 1, -1 to 1, ...
        - Nonzero is 'activated'
        - Examples
            + Linear
            + Rectified linear units (ReLU) and Leaky ReLU
            + Sigmoid
            + Tanh and hard Tanh
            + Softmax and hierarchical softmax
            + Softplus
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

**Model Performance**
- Overfitting and quality of fit
    + Bias vs variance
    + Noise
    + Outliers
- Performance metric selection
    + Precision vs recall
    + ROC and AUC
- Performance metric tradeoffs
    + Cost of false positives vs false negatives
- Inability to achieve desired performance level
    + Intractability
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

**Model Complexity**
- Subset selection
    + Best subset selection
    + Stepwise selection (forward and backward)
- Shrinkage and regularization
    + Ridge regression
    + The Lasso
- Dimension reduction
    + PCA
    + Partial least squares
- Tree methods (e.g., pruning, ...)
- Feature selection and engineering
    + Collinearity, multicollinearity, correlation, ...
    + Confounding variables
    + Missing features

**Model Training, Validation, Tuning, and Optimization**
- Resampling
    + Cross-validation
    + Bootstrap
- Hyperparameter tuning
    + Example categories <sup>4</sup>
        * Layer size
        * Magnitude (momentum, learning rate)
        * Regularization (dropout, drop connect, L1, L2)
        * Activations (and activation function families)
        * Weight initialization strategy
        * Loss functions
        * Settings for epochs during training (mini-batch size)
        * Normalization scheme for input data (vectorization)
- Grid search
- Ensemble methods
- Bagging and boosting
- Kernel selection (e.g., SVM)

**Data and Data Sources**
- Balanced vs imbalanced data
    + Equal proportion of target values
- Data availability, amount, depth, and quality
    + Curse of dimensionality
    + Small data sets
        * Selecting models that excel with small data sets
    + Sparse data
    + Data veracity
    + Signal to noise ratio
- Curse of dimensionality
    + Exponential increase in data needed to prevent loss of predictive power
    + Data must fill as much as the feature space as possible
- Working with large datasets <sup>1</sup><sup>,</sup><sup>2</sup>
    + Partition data vs partition execution
    + Software-specific memory allocation
    + Full dataset vs subset (down sampling, e.g., 1/10 to 1/100)
    + Local development vs remote compute resources (e.g., AWS) with lots of RAM
    + Data format transformation
    + Iterative learning
        * Stochastic gradient descent
        * Mini-batch
        * KNN
    + Streaming machine learning
    + Progressively loading
    + Direct database connection for machine learning
    + Big data platformes (e.g., Spark and MLLib)
    + Parallel and distributed computing and associated communications cost and complexity
    + Parameter server
- Data types
    + Structured data
    + Unstructured data
    + Semi-structured data
    + Streaming data
    + Batch data
- Data acquisition, ingestion, and processing
    + Disparate data sources

**Computing and Infrastructure Requirements**
- CPU processing speed and computing time
    + Data processing, model training, and prediction speed
- CPU vs GPU
- System memory (RAM)

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

**General**
- Model performance vs interpretability and explainability
    + Black box vs non-black box algorithms
- Model complexity vs simplicity (i.e., parsimony)
- Statistical, algorithmic, and cognitive biases
- Generalization vs representation

**AI and Machine Learning**
- AI limitations
    + Unsupervised learning to some extent
    + Solving multiple problems at once
- Statistical, algorithmic, and cognitive biases
    + Sample or selection bias
    + Confirmation bias
    + Algorithmic bias <sup>3</sup>
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

## References

1. [7 Ways to Handle Large Data Files for Machine Learning](https://machinelearningmastery.com/large-data-files-machine-learning/)
2. [Beyond Distributed ML Algorithms: Techniques for Learning from Large Data Sets](https://iwringer.wordpress.com/2015/10/06/techniques-for-learning-from-large-amounts-of-data/)
3. [Joy Buolamwini - TED Talk](https://www.ted.com/talks/joy_buolamwini_how_i_m_fighting_bias_in_algorithms)
4. [Deep Learning by Josh Patterson and Adam Gibson - O'Reilly](https://www.amazon.com/Deep-Learning-Practitioners-Josh-Patterson-ebook/dp/B074D5YF1D/ref=mt_kindle?_encoding=UTF8&me=)
