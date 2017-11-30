## Goals

- Derive discoveries, insights, and value from data collections
- Turning data into actions, including real-time
- Build data products providing actionable information while abstracting away technical details, data, and analytics
- Question discovery
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
- Grow data economy wealth
- Shifting between deductive (hypothesis-based) and inductive (pattern- based) reasoning (ref. Booz, Allen, Hamilton)
    + Deductive
        * Formulate hypotheses about relationships and underlying models
        * Carry out experiments with the data to test hypotheses and models
    + Inductive
        * Exploratory data analysis to discover or refine hypotheses
        * Discover new relationships, insights and analytic paths from the data
- Business-level goals
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
- Self-service (ad-hoc) analytics and insights
- Personalize customer experiences
- Health diagnosis
- Enterprise performance management (EPM)
- Information discovery

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

- Overfitting and quality of fit
    + Bias vs variance
    + Noise
    + Outliers
- Model performance vs interpretability
    + Black box vs non-black box algorithms
- Model performance
    + Performance metric selection
    + Inability to achieve desired performrmance level
    + Collinearity, multicollinearity, correlation, ...
    + Confounding variables
    + Missing features
    + Global vs local minima
- Model complexity reduction
    + Subset selection
        * Best subset selection
        * Stepwise selection (forward and backward)
    + Shrinkage and regularization
        * Ridge regression
        * The Lasso
    + Dimension reduction
        * PCA
        * Partial least squares
    + Tree methods (e.g., pruning, ...)
- Model complexity vs simplicity (i.e., parsimony)
    + Dimensionality reduction
    + Regulariziation
    + Feature selection and engineering
    + Model selection
    + Resampling methods
- Different statistical, algorithmic, and cognitive biases
- Unsupervised learning limitations for AI
- Balanced vs imbalanced data
    + Equal proportion of target values
- Feature selection and engineering
    + Collinearity, multicollinearity, correlation, ...
    + Confounding variables
    + Missing features
- Curse of dimensionality
    + Exponential increase in data needed to prevent loss of predictive power
    + Data must fill as much as the feature space as possible
- Model selection
    + Parametric vs non-parametric
        * Parametric examples
            - Simple and multiple linear regresion
        * Non-parametric examples
            - Decision trees
            - KNN - k nearest neighbors
            - SVMs
- Model tuning and performance optimization
    + Hyperparameter tuning
    + Grid search
    + Ensemble methods
    + Bagging and boosting
- Resampling
    + Cross-validation
    + Bootstrap
- Computing resources and power
    + CPU processing speed and computing time
    + CPU vs GPU
    + System memory (RAM)
- Working with large datasets <sup>1</sup><sup>,</sup><sup>2</sup>
    + Partition data vs partition execution
    + Software-specific memory allocation
    + Full dataset vs subset (down sampling, e.g., 1/10 to 1/100)
    + Local development vs remote compute resources (e.g., AWS) with lots of RAM
    + Data format transformation
    + Iterative learning (e.g., stochastic gradient descent)
    + Streaming machine learning
    + Progressively loading
    + Direct database connection for machine learning
    + Big data platformes (e.g., Spark and MLLib)
    + Parallel and distributed computing and associated communications cost and complexity
    + Parameter server
- AI limitations
    + Unsupervised learning to some extent
    + Solving multiple problems at once
- AI and machine learnng in production
    + Deploying and maintaining
    + Scalability
    + Online vs offline learning
- Statistical, algorithmic, and cognitive biases
    + Sample or selection bias
    + Confirmation bias
    + Algorithmic bias <sup>3</sup>
- AI and machine learning process
    + CRISP-DM, etc.
- AI and machine learning automations and future advancements
    + Automated learning (AutoML)
        * Auto-sklearn, TPOT, ...


## Real-world Applications and Vendors by Category

- Recommender systems
    + Netflix
        * Increase engagement, retention, and revenues
        * Examples
            - "Because you watched ..."
            - "Top picks for ..."
            - Recommendations by category
                + Trending Now
                + Neflix originals
                + TV Documentaries
    + Amazon
        * Increase average order size and therefore sales (studies show between 5.9 to 30%)
        * Examples
            - "Customers who bought this item also bought"
            - "Customers who viewed this item also viewed"
            - "What other items do customers buy after viewing this item?"
            - "Recommendations for you in ..." (e.g., "Recommended for You in Kindle Books")
            - "New for you"
            - "Inspired by your shopping trends"
            - "Inspired by your Wish List"
    + Robo-advisors and portfolio rebalancing
        * [Weathfront](https://www.wealthfront.com/)
        * [Betterment](https://www.betterment.com/)
    + Spotify
        * [Daily Mix](https://support.spotify.com/us/using_spotify/search_play/daily-mix/)
    + Personalized news feeds, including Facebook
- Prediction and Classification
    + Many diseases or issues, including stroke, cancer, ...
        * Cancer detection using cell-free genomes
        * Cardiovascular events prediction (e.g., heart attack, stroke)
        * Companies
            - [Google DeepMind](https://deepmind.com/)
            - IBM's [Watson](https://www.ibm.com/watson/) (Watson for Oncology)
            - Others
                - [Freenome](https://www.freenome.com/)
                - [CureMetrix](http://curemetrix.com/)
    + Spam for email
    + Smart email categorization (Gmail)
        * Primary, social, and promotion inboxes, as well as labeling emails as important
    + Stock market predictions and algorithmic trading
        * Companies
            - [Kavout](https://www.kavout.com/)
            - [Sentient](http://www.sentient.ai/)
            - [Genotick](http://genotick.com/)
            - [Numerai](https://numer.ai/)
            - [QPLUM](https://www.qplum.co/)
    + Crime
        * Who, Type, and location
        * Based on previous crime and social media activity
        * Companies
            - [BRS Labs AISight](https://www.genetec.com/solutions/resources/brs-labs-aisight-technology-and-omnicast-integration)
    + Suicide risk
        * Based on a lot of different risk factors
        * Companies
            - [Facebook](https://research.fb.com/category/facebook-ai-research-fair/)
            - [Instagram](https://www.instagram.com/)
            - [Cogito](https://www.cogitocorp.com/)
    + Uber's ETA
    + Credit decisions
        * Companies
            - [Underwrite.ai](http://www.underwrite.ai/)
    + Agriculture - predicting crop yields
        * Companies
            - [Descartes Lab](http://www.descarteslabs.com/)
            - [Stanford's Sustainability and Artificial Intelligence Lab](http://sustain.stanford.edu/)
- Computer vision and recognition
    + Computer vision
        * Components
            - Recognition (e.g., objects)
            - Motion analysis
            - Scene reconstruction
            - Image restoration
        * Examples
            - Manufacturing 
                + Inspections
                + Quality control
                + Assembly line
            - Visual surveillance
                + Companies
                    * [BRS Labs AISight](https://www.genetec.com/solutions/resources/brs-labs-aisight-technology-and-omnicast-integration)
            - Navigation, including autonomous vehicles
                + Land, water, and space
            - Medical image processing and diagnosis
            - Military
                + Detection of enemy solidiers and vehicles
                + Missle guidance
            - Drones
                + Inspection (pipelines), surveillance, exploration (buildings), and protection
                + Companies
                    * [Digital Signal](http://www.digitalsignalcorp.com/)
                    * [Shield.ai](http://shield.ai/)
            - Item recognition
                + Companies
                    * [Amazon Go](https://www.amazon.com/b?node=16008589011)
    + Recognition
        * [Shazam](https://www.shazam.com/)
        * Wine
            - Companies
                + [Delectable](https://delectable.com/)
                + [Vivino](https://www.vivino.com/)
        * Facebook photo recognition (highlights faces and suggests friends to tag)
        * Speech/Voice to text (faster to talk than to type acc'g to Stanford)
            - Companies
                + [Google Cloud Speech API](https://cloud.google.com/speech/)
        * Text to speech
            - Companies
                + [Amazon Polly](https://aws.amazon.com/polly/)
        * Video
            - Companies
                + [Clarifai](https://www.clarifai.com/)
                + [Google Cloud Video Intelligence](https://cloud.google.com/video-intelligence/)
        * OCR
            - Mobile app check deposits and uploading receipts
            - Post office address recognition
        * Object recognition
            - Companies
                + [Pinterest](https://medium.com/@Pinterest_Engineering) (then used to recommend other pins)
        * Image
            - Companies
                + [Clarifai](https://www.clarifai.com/)
                + [Captricity](http://captricity.com/)
                + [Google Cloud Vision API](https://cloud.google.com/vision/)
                + [Amazon Rekognition](https://aws.amazon.com/rekognition/)
- Clustering and anomaly detection
    + Clustering
    + Anomaly detection
        * Manufacturing
        * Data security
            - Companies
                + [Cylance](https://www.cylance.com/en_us/home.html)
                + [Darktrace](https://www.darktrace.com/)
        * Personal security (security screenings at airports, stadiums, concerts, and other venues)
        * Law enforcement
        * Application performance
        * Credit card fraud detection
- Natural language
    + Smart personal assistants
        * Companies
            - [Alexa](https://developer.amazon.com/alexa)
            - [Google Assistant](https://assistant.google.com/)
            - [Siri](https://www.apple.com/ios/siri/)
        * Uses
            - Internet searches and answer questions
            - Set reminders
            - Integrate with your calendar
                + Make appointments
            - Receive sports, news, and finance updates
            - Create to-do lists
            - Order items online
            - Use services (e.g., order an Uber)
            - Play music
            - Play games
            - Smart home integration
    + NLG - computer generated reports and news
        * Summarizing documents
        * Story telling
        * Sports recaps
        * Companies
            - [Narrative Science](https://www.narrativescience.com/)
    + NLP and language translation
        * Voicemail transcripts
        * eDiscovery
        * Companies
            - [Google Natural Language API](https://cloud.google.com/natural-language/)
            - [Google Cloud Translation API](https://cloud.google.com/translate/)
            - [Textio](https://textio.com/) for writing optimal job descriptions
    + NLU and Chatbots
        * Shopping
        * Errands
        * Day to day tasks
        * Companies
            - [x.ai](https://x.ai/) (personal assistant)
            - [MindMeld](https://www.mindmeld.com/)
            - [Google Inbox Smart Reply](https://blog.google/products/gmail/save-time-with-smart-reply-in-gmail/)
            - [Amazon Lex](https://aws.amazon.com/lex/), includes Automatic speech recognition (ASR)
    + Smart instant messaging
        * Companies
            - [Google Allo](https://allo.google.com/) smart messaging app (https://allo.google.com/)
- Marketing and Sales
    + Marketing
        * Market basket analysis > location and promotions of items
        * Cohort analysis and segmentation > targeted marketing
        * Customer churn prediction > churn prevention
        * Customer lifetime value forecasting > future business value and predicting growth
        * Targeted and personalized advertising
        * Companies
            - [Appier](https://www.appier.com/)
            - [Voyager Labs](http://voyagerlabs.co/)
    + Sales and forecasting
        * Revenue and growth
- Other
    + Google search
    + Autonymous vehicles (Business insider)
        * Reduce accidents and related injuries and death
        * Improved traffic (via ridesharing and smart traffic lights) and fuel efficiency
        * Reduced carbon emissions
        * Faster commutes and travel time
        * Get your time back in the vehicle to do what you want
        * Efficient ride-sharing
        * Companies
            - [Zoox](http://zoox.com/)
            - [Nauto](http://www.nauto.com/)
            - [nuTonomy](http://nutonomy.com/)
    + Home monitoring, control, and security
        * Companies
            - [Flare](https://buddyguard.io/)
    + Voice-controled robotics
    + Photo-realistic pictures generation from text or sketches
        * [NYU article](http://cds.nyu.edu/astronomers-explore-uses-ai-generated-images-using-ai-generating-imagess/)
    + Music generation
        * Companies
            - [Jukedeck](https://www.jukedeck.com/)
    + Movie and script generation
    + Automatically generated software code
        * Companies
            - [DeepCoder](https://openreview.net/pdf?id=ByldLrqlx) (Microsoft and Cambridge)
    + Authentication without passwords (using mobile phone that knows it's you)
        * Companies
            - [TypingDNA](https://typingdna.com/)
    + Customer support
        * Companies
            - [DigitalGenius](https://www.digitalgenius.com/)
    + Optimized directions and routes
    + Plagiarism Checkers
    + Robo-readers and graders
    + Virtual reality
    + Gaming
    + [Zillow’s](https://www.zillow.com/zestimate/) “zestimate” feature, which estimates the price of homes
    + Medical/Health
        * Companies
            - [BenevolentAI](http://benevolent.ai/)
    + Sales
        * Companies
            - [InsideSales.com](https://www.insidesales.com/)

## References

1. [7 Ways to Handle Large Data Files for Machine Learning](https://machinelearningmastery.com/large-data-files-machine-learning/)
2. [Beyond Distributed ML Algorithms: Techniques for Learning from Large Data Sets](https://iwringer.wordpress.com/2015/10/06/techniques-for-learning-from-large-amounts-of-data/)
3. [Joy Buolamwini - TED Talk](https://www.ted.com/talks/joy_buolamwini_how_i_m_fighting_bias_in_algorithms)
