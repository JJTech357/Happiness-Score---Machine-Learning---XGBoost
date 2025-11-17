# 🧠 Predicting Happiness: A Well-being Regression Model

## 1\. Project Title and Description

This project develops a machine learning model to predict an individual's **"happiness score"** based on a comprehensive life and well-being survey.

Using **Multiple Linear Regression (MLR)** in **R**, the model analyzes demographic, socio-economic, and psychological factors to understand and quantify their impact on the final continuous happiness score. The goal is to build an accurate and interpretable model that establishes a baseline for predicting individual well-being.

-----

## 2\. Table of Contents

1.  [Project Title and Description](https://www.google.com/search?q=%231-project-title-and-description)
2.  [Table of Contents](https://www.google.com/search?q=%232-table-of-contents)
3.  [Installation Instructions](https://www.google.com/search?q=%233-installation-instructions)
4.  [Dataset Description](https://www.google.com/search?q=%234-dataset-description)
5.  [Modeling Approach](https://www.google.com/search?q=%235-modeling-approach)
6.  [Results and Evaluation](https://www.google.com/search?q=%236-results-and-evaluation)
7.  [Usage](https://www.google.com/search?q=%237-usage)
8.  [Project Structure](https://www.google.com/search?q=%238-project-structure)
9.  [Contributing Guidelines](https://www.google.com/search?q=%239-contributing-guidelines)
10. [License](https://www.google.com/search?q=%2310-license)

-----

## 3\. Installation Instructions

This project is implemented in R and executed within a Jupyter Notebook environment (using an R kernel).

### Prerequisites

  * **R Environment:** Ensure R (version 3.6.3 or newer is recommended) is installed.
  * **Jupyter:** Install Jupyter Notebook/JupyterLab.
  * **IRKernel:** Install and configure the R kernel for Jupyter:
    ```bash
    # In R console:
    install.packages('IRkernel')
    IRkernel::installspec()
    ```

### Required R Packages

Install the following packages in your R environment:

```r
install.packages(c(
    "dplyr",
    "tidyr",
    "performance",
    "visreg",
    "fastDummies",
    "caret",
    "nnet",
    "xgboost"
))
```

### Running the Notebook

1.  Clone the repository:
    ```bash
    git clone [Your Repository URL]
    cd [Your Repository Name]
    ```
2.  Place the required data files (`regression_train.csv` and `regression_test.csv`) into the root directory.
3.  Launch Jupyter Notebook:
    ```bash
    jupyter notebook
    ```
4.  Open and run the notebook: `FIT5197_FinalAssessment (1).ipynb`. Select the **R** kernel.

-----

## 4\. Dataset Description

### Source Files

The model uses two primary datasets, assumed to be provided separately from the repository:

  * `regression_train.csv`: Training data with all predictor variables and the target variable (`happiness`).
  * `regression_test.csv`: Test data with only predictor variables, used for prediction.

### Features

The dataset consists of over 40 features derived from survey questions, including:

| Feature Type | Examples | Description |
| :--- | :--- | :--- |
| **Target Variable** | `happiness` | A continuous numerical score representing the individual's happiness level. |
| **Categorical** | `gender`, `income`, `whatIsYourHeight...` | Factors with multiple levels (e.g., income ranges, height bands). |
| **Ordinal/Numeric** | `alwaysAnxious`, `alwaysStressed`, `iFindBeautyInSomeThings` | Self-assessment scores on a scale, typically ranging from **-2** to **+2**. |

### Preprocessing

  * **Handling Categorical Variables:** The primary Multiple Linear Regression model (`lm`) automatically handles factor variables by creating dummy (one-hot) encoded features, while avoiding multicollinearity by dropping the first level (reference category).
  * **Feature Mapping (Exploration):** The notebook includes code blocks for manual ordinal encoding of variables like `income` and `height` into a single numeric value for exploratory analysis and alternative models.

-----

## 5\. Modeling Approach

### Algorithm

The core model for happiness score prediction is **Multiple Linear Regression (MLR)**.

### Rationale

MLR was chosen as the initial approach due to its **interpretability**. By examining the model coefficients and their statistical significance (p-values), we can directly quantify the direction and magnitude of the relationship between each predictor and the `happiness` score.

### Model Development

1.  **Full Model (`lm.fit`):** An initial model was trained using all available features (both categorical and numeric).
    ```r
    lm.fit <- lm(happiness ~ ., data=train)
    ```
2.  **Feature Selection:** Predictors with a statistical significance of $p \le 0.01$ were identified to understand the core drivers of happiness.
3.  **Model Comparison:** The full model was compared against a simpler model that excluded all categorical variables to determine the value added by socio-economic and demographic data.

-----

## 6\. Results and Evaluation

### Performance Metrics

The model was evaluated using **Root Mean Squared Error (RMSE)**, which measures the average magnitude of the errors.

| Model | Description | RMSE (Training Data) |
| :--- | :--- | :--- |
| **`lm.fit` (Full MLR)** | Includes all features (categorical and numeric) | **4.250579** |
| `lm.fit2` (Numeric Only) | Excludes all categorical features | 4.693674 |

The **Full MLR Model** provided the superior performance, indicating that the categorical features (especially income) are crucial predictors.

### Key Predictors

Based on the absolute t-value (which indicates the strength of the association), the strongest predictors for the `happiness` score in the full model were predominantly income-related variables:

1.  `income80k - 120k`
2.  `income50k - 80k`
3.  `income200k above`
4.  `income20k - 50k`
5.  `income15k - 20k`

-----

## 7\. Usage

This model can be used as a baseline for predicting continuous well-being scores in similar survey datasets. To adapt the model:

1.  **Feature Engineering:** If using a new dataset, ensure categorical features are correctly recognized as factors in R to allow for implicit dummy encoding by the `lm()` function.
2.  **Retraining:** Use the same `lm(happiness ~ ., data=new_train_data)` structure for initial training.
3.  **Advanced Models:** This MLR approach provides a highly interpretable result. For potentially better performance, the methodology can be extended to implement more complex models like **XGBoost (eXtreme Gradient Boosting)** or **Neural Networks (nnet)**, for which required packages are included in the setup.

-----

## 8\. Project Structure

```
.
├── FIT5197_FinalAssessment (1).ipynb  # R Jupyter Notebook with code, analysis, and models.
├── 32455038-final-assignment.pdf      # Detailed report/explanation of the project and results.
├── regression_train.csv               # (Data - not included in repo, required to run)
├── regression_test.csv                # (Data - not included in repo, required to run)
└── README.md                          # This file.
```

-----

## 9\. Contributing Guidelines

Contributions are welcome\! Please open an issue first to discuss the desired changes or enhancements.

-----
