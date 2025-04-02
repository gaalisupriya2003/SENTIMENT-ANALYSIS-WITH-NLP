# SENTIMENT-ANALYSIS-WITH-NLP

**COMPANY**: CODTECH IT SOLUTIONS

**NAME**: GALI SUPRIYA

**INTERN ID**:CT12WJVC

**DOMAIN**: MACHINE LEARNING

**BATCH DURATION**:JANUARY 5TH TO APRIL 5TH,2025

**MENTOR NAME**:NEELA SANTHOSH 

# Sentiment Analysis on Restaurant Reviews Using TF-IDF and Logistic Regression

Sentiment analysis is a *natural language processing (NLP) task* that classifies text as positive, negative, or neutral. This project performs *sentiment analysis* on a *restaurant reviews dataset* to determine customer sentiments using *TF-IDF vectorization* and *Logistic Regression*. It helps restaurant owners understand customer opinions and improve services based on feedback.  

## Tools and Technologies Used  
- *Programming Language:* Python  
- *Libraries:* NLTK, scikit-learn, Pandas, NumPy, Matplotlib, Seaborn  
- *Dataset:* `restaurant_reviews.csv`from kaggle containing text reviews and corresponding sentiment labels (positive/negative)  
- *Machine Learning Model:* Logistic Regression for binary classification  
- *Feature Extraction:* TF-IDF (Term Frequency-Inverse Document Frequency) to convert text into numerical vectors  

## Dataset and Preprocessing  
The dataset consists of *restaurant reviews* labeled as positive or negative. The text data is preprocessed with the following steps:  
1. *Lowercasing:* Converts text to lowercase to maintain consistency.  
2. *Removing Punctuation and Numbers:* Eliminates unnecessary characters to clean data.  
3. *Tokenization:* Splits sentences into individual words using *NLTK*.  
4. *Stopword Removal:* Eliminates common words (e.g., “the,” “is”) that do not contribute to sentiment.  
5. *TF-IDF Vectorization:* Converts text into a *numerical matrix of 2,000 features*, capturing word importance based on frequency.  

## Model Training and Evaluation  
The dataset is split into *80% training* and *20% testing. The **Logistic Regression* model is trained on the TF-IDF transformed data. *Hyperparameters* are tuned for better accuracy.  

### Performance Metrics:  
1. *Accuracy:* The model achieves between *81% accuracy*.
2. *Confusion Matrix:* A heatmap visualizing correct vs. incorrect predictions. 
3. *Classification Report:* Displays precision, recall, and F1-score for sentiment classes.  

### Observations:  
- The model performs well in identifying positive and negative sentiments.  
- Some misclassifications occur due to *ambiguous or mixed-sentiment reviews*.  
- Increasing dataset size and fine-tuning *TF-IDF parameters* can improve performance.  

## Applications of Sentiment Analysis  
1. *Restaurant Management:* Helps analyze customer feedback, identify service issues, and enhance dining experiences.  
2. *Online Review Platforms:* Automatically filters and classifies reviews as positive or negative for better user experience.  
3. *Brand Reputation Monitoring:* Tracks *public perception* of restaurants across social media and review sites.  
4. *Automated Customer Support:* AI-driven chatbots can use sentiment analysis to prioritize negative reviews for quicker response.  
5. *Market Research:* Helps businesses understand *customer preferences* and adjust services accordingly.  

This project provides a *robust sentiment analysis system* for restaurant reviews, enabling better decision-making based on customer opinions.
