# Anemia Diagnosis Using CBC Data

## Overview
This project focuses on the automated classification of anemia types using Complete Blood Count (CBC) data. By leveraging machine learning techniques, specifically a stacking ensemble classifier, the model achieves high accuracy in diagnosing various anemia types.

## Dataset
The dataset includes key CBC parameters and is labeled with specific anemia diagnoses. The parameters are:
- **WBC**: White Blood Cell count
- **LYMp**: Lymphocytes percentage
- **NEUTp**: Neutrophils percentage
- **LYMn**: Lymphocytes number
- **NEUTn**: Neutrophils number
- **RBC**: Red Blood Cell count
- **HGB**: Hemoglobin
- **HCT**: Hematocrit
- **MCV**: Mean Corpuscular Volume
- **MCH**: Mean Corpuscular Hemoglobin
- **MCHC**: Mean Corpuscular Hemoglobin Concentration
- **PLT**: Platelets count
- **PDW**: Platelet Distribution Width
- **PCT**: Procalcitonin
- **Diagnosis**: Type of anemia

## Data Visualization
- **Count Plot of Diagnoses**: Shows the distribution of each diagnosis category.
- **Distribution Plots**: Histograms and KDE plots for each CBC parameter.
- **Box Plots**: Display the distribution of each CBC parameter.
- **Pair Plot**: Visualizes the relationships between selected features.

## Machine Learning
### Model Training
- **Features**: All CBC parameters.
- **Target**: Anemia type (Diagnosis).
- **Train/Test Split**: 80% training, 20% testing.

### Ensemble Learning
- **Estimators**:
  - Random Forest Classifier
  - K-Nearest Neighbors Classifier
  - Gradient Boosting Classifier
- **Final Estimator**: Logistic Regression
- **Cross-validation**: 10-fold

### Model Evaluation
- **Accuracy**: 98%
- **Confusion Matrix**: Visualized using a heatmap.
- **Classification Report**: Includes precision, recall, and F1-score for each class.

## Results
The model achieves high accuracy in predicting anemia types, demonstrating its effectiveness for this classification task. The final predictions are saved to a CSV file.

## Files
- **diagnosed_cbc_data_v4.csv**: Original dataset.
- **df_test.csv**: Test dataset.
- **Prediction.csv**: Test dataset with predictions.

## How to Run
1. Ensure you have the necessary libraries:
    ```bash
    pip install pandas matplotlib seaborn scikit-learn
    ```
2. Load the data and preprocess:
    ```python
    import pandas as pd
    df = pd.read_csv('diagnosed_cbc_data_v4.csv')
    ```
3. Visualize the data:
    ```python
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Example: Count Plot of Diagnoses
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Diagnosis', data=df)
    plt.title('Count of Each Diagnosis Category')
    plt.xlabel('Diagnosis')
    plt.ylabel('Count')
    plt.xticks(rotation=90)
    plt.show()
    ```
4. Train the model:
    ```python
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression

    x = df.drop(columns=['Diagnosis'])
    y = df['Diagnosis']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=43)

    estimators = [
        ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
        ('knn', KNeighborsClassifier(n_neighbors=10)),
        ('gbdt', GradientBoostingClassifier())
    ]

    clf = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(),
        cv=10
    )
    clf.fit(x_train, y_train)
    ```
5. Evaluate the model:
    ```python
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

    y_pred = clf.predict(x_test)
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))
    print('Classification Report:')
    print(classification_report(y_test, y_pred))
    ```

6. Save predictions:
    ```python
    ndf = pd.read_csv('df_test.csv')
    ndf['prediction'] = y_pred
    ndf.to_csv('Prediction.csv', index=False)
    ```

## License
This project is licensed under the MIT License.
