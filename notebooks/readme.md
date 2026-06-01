# Multi-Hospital Patient Journey Bottleneck Miner

## Problem
Healthcare machine learning projects usually focus on predicting diagnosis. This project takes a different direction by mining where delays happen in the patient journey, from admission to treatment and finally discharge, so hospitals can identify operational bottlenecks, reduce Length of Stay (LoS), and improve efficiency.

## Dataset
This project uses the Kaggle Healthcare Dataset by prasad22, containing 55,500 patient admission records collected across multiple hospitals.

## Tech Stack
PySpark 3.5.1 (DataFrames, Spark SQL, MLlib), Python 3.11.9, Jupyter Notebook, Matplotlib, Seaborn, scikit-learn, Dash, and Plotly.

## Pipeline
1. Load the raw CSV dataset into a Spark DataFrame.  
2. Clean the data, cast dates, and compute Length of Stay (LoS) using admission and discharge dates. 
3. Engineer additional features such as age bands, cost per day, hospital average LoS, condition average LoS, and admission-type average LoS. 
4. Perform Spark SQL aggregation to rank hospitals by bottleneck severity using average and percentile-based LoS statistics.  
5. Use a window function to flag the top 10% slowest stays within each hospital as bottleneck cases.  
6. Build a dual-task ML pipeline where Stage 1 predicts LoS and Stage 2 predicts diagnosis from delay-related features.
7. Apply KMeans clustering with \(k=3\) to group hospitals into Efficient, Average, and Bottleneck clusters.

## Key Findings
The analysis ranked 39 hospitals with sufficient admission volume and identified the worst bottleneck hospitals by average stay duration. 
The top three bottleneck hospitals were **Brown PLC** with average LoS 21.14 days and P90 LoS 30 days, 
**Inc Williams** with average LoS 19.04 days and P90 LoS 29 days,
 and **Smith PLC** with average LoS 17.64 days and P90 LoS 27 days.

The worst condition and admission-type combination was **Cancer + Emergency**, with an average LoS of 15.99 days and P90 LoS of 28 days. 
Other high-delay combinations included **Asthma + Urgent** at 15.77 days and **Asthma + Elective** at 15.74 days, showing that both condition and admission route contribute to bottlenecks.

The main GBT regressor achieved **RMSE = 3.94 days**, **MAE = 1.80 days**, and **R² = 0.7934**, meaning it explained about 79.3% of the variance in Length of Stay on this synthetic dataset. 
In the second stage, logistic regression for diagnosis-from-delay achieved **Accuracy = 0.5605** and **F1-score = 0.4936**.

KMeans clustering separated hospitals into three operational groups: **22 Bottleneck hospitals**, 
**9 Average hospitals**, and **8 Efficient hospitals**. 
Their average LoS values were 16.65, 15.12, and 13.75 days respectively, which supports the interpretation that hospital-level patterns strongly affect delay outcomes.

Feature importance from the GBT model showed that delay was driven most strongly by **gender (0.0116)**, **hospital average LoS (0.0103)**, **admission-type average LoS (0.0093)**, **age band (0.0071)**, and **age (0.0070)**. This suggests that both patient-level and hospital-level features matter, but hospital baseline delay remains one of the strongest signals.

## Output Files
All outputs are saved in the `./output/` folder.

- `clean_journey.parquet` — cleaned dataset with engineered features.  
- `hospital_bottleneck/` — ranked hospitals by average LoS and percentile metrics  
- `condition_admission_bottleneck/` — condition × admission-type bottleneck summary  
- `bottleneck_drivers/` — grouped delayed cases by hospital, condition, and admission type 
- `flagged_cases/` — flagged top-delay patient journeys.  
- `hospital_clusters/` — KMeans clustering results with hospital labels. 
- `los_predictions/` — predicted Length of Stay outputs. 
- `diagnosis_from_delay/` — diagnosis predictions using delay-informed features.

## Visualizations
The project includes static charts and an interactive dashboard for presenting the results.

- Hospital bottleneck bar chart  
- Condition × admission-type heatmap  
- Interactive dashboard built with Dash and Plotly

## Limitations
The dataset is synthetic, so clinical relationships may not fully reflect real hospital behavior .
 It also contains date-level information only, without fine-grained timestamps for admission, treatment start, lab delays, or discharge processing, which limits deeper bottleneck tracing.

## Future Work
Future improvements can make the project more realistic and more useful for hospital operations.

1. Integrate real EHR timestamps for finer-grained bottleneck detection inside each patient stay  
2. Add treatment-stage timing such as lab turnaround, medication delay, and discharge processing  
3. Extend the cost analysis using LoS and billing features to estimate financial loss from delays  
4. Deploy the trained pipeline as a web API or hospital dashboard for real-time monitoring.

## How to Run
Use Python 3.11.9 with a virtual environment, install the required libraries, and run both notebooks in Jupyter.

```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate environment
.\venv\Scripts\activate

# 3. Install dependencies
pip install pyspark==3.5.1 pandas matplotlib seaborn jupyter findspark pyarrow scikit-learn dash plotly

# 4. Start Jupyter
jupyter notebook
```

Then run:
- `bottleneck_miner-2.ipynb` for Spark analysis  
- `bottle_neck_display.ipynb` for the dashboard

## Team
[Inshrah Alam  L1F22BSCSc0384
M.Zunair Arif L1F22BSCS0957
M.Raffay Abbas L1F22BSCS0968
Tayyab Iqbal L1S22BSCS0019]


## Dataset Link
[Kaggle Healthcare Dataset](https://www.kaggle.com/datasets/prasad22/healthcare-dataset)