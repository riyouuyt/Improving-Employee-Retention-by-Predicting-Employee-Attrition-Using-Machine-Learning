# Improving-Employee-Retention-by-Predicting-Employee-Attrition-Using-Machine-Learning

<div align="center">
  <img src="https://github.com/riyouuyt/Improving-Employee-Retention-by-Predicting-Employee-Attrition-Using-Machine-Learning/assets/122600889/44c5d67c-bcdd-45ca-8a69-13225c1fd995" alt="Image" width="780" height="400" />
</div>



## Project Description
In this project, we aim to tackle the critical issue of employee attrition by leveraging the power of machine learning. Employee retention is not only crucial for the well-being of the workforce but also for the success and stability of any organization. We will go through the following steps to achieve this:

## Objective 🎯
The objective of this project is to address the challenge of employee attrition within organizations by applying machine learning techniques to predict potential resignations. By leveraging historical data and predictive models, the goal is to empower HR departments with insights to proactively identify at-risk employees and take preemptive measures to improve retention rates.

## Libraries and Tools
For this project, you will need a variety of libraries and tools, including but not limited to:

- **Python**
- **NumPy**
- **Pandas**
- **Scikit-Learn**
- **Matplotlib**
- **Seaborn**
- **NLTK** or **SpaCy** (for NLP)
- **Jupyter Notebooks**
- **GitHub** or other version control systems


## Goals 
- Create an annual report on employee count changes to visualize trends and patterns.
- Perform an in-depth analysis of resignation reasons through sentiment analysis and thematic identification.
- Build and deploy machine learning models capable of accurately predicting employee resignations.
- Provide a user-friendly interface for HR personnel to access and utilize the predictive models effectively.
- Present findings and insights to stakeholders to aid in decision-making and policy formulation.

## Data Understanding

### Dataset Information
This dataset comprises 287 rows and 25 columns, occupying approximately 56.2+ KB in memory.

### Columns Overview
The dataset contains various features:

- **Rows:** `287`
- **Columns:** `25`
- **Memory Usage:** `56.2+ KB`

### Key Column Information
The dataset includes the following essential columns:

1. **`Username`:** Unique employee username.
2. **`EnterpriseID`:** ID within the enterprise.
3. **`StatusPernikahan`:** Marital status.
4. **`JenisKelamin`:** Gender.
5. **`StatusKepegawaian`:** Employment status.
6. **`Pekerjaan`:** Job position.
7. **`JenjangKarir`:** Career level.
8. **`PerformancePegawai`:** Performance rating.
9. **`AsalDaerah`:** Region of origin.
10. **`HiringPlatform`:** Platform used for hiring.
11. **`SkorSurveyEngagement`:** Engagement score.
12. **`SkorKepuasanPegawai`:** Employee satisfaction score.
13. **`JumlahKeikutsertaanProjek`:** Number of projects participated in.
14. **`JumlahKeterlambatanSebulanTerakhir`:** Recent monthly delay count.
15. **`JumlahKetidakhadiran`:** Absence count.
16. **`NomorHP`:** Employee's phone number.
17. **`Email`:** Employee's email address.
18. **`TingkatPendidikan`:** Educational level.
19. **`PernahBekerja`:** Previous employment history.
20. **`IkutProgramLOP`:** Participation in a specific program (LOP).
21. **`AlasanResign`:** Reason for resignation (if applicable).
22. **`TanggalLahir`:** Date of birth.
23. **`TanggalHiring`:** Date of hiring.
24. **`TanggalPenilaianKaryawan`:** Date of employee evaluation.
25. **`TanggalResign`:** Date of resignation for resigned employees.


## Data Preprocessing 📦

### Handling Missing Values

Before we start modelling, there are few step that need to be consider more and that is checking the missing values since machine learning will not process any data given if there's still empty data, so in order having a perfect model
let's check the percentage and total of missing values on our dataset.

- `IkutProgramLOP:` 89% (258 missing values)
- `AlasanResign:` 23% (66 missing values)
- `JumlahKetidakhadiran:` 2% (6 missing values)
- `SkorKepuasanPegawai:` 1.7% (5 missing values)
- `JumlahKeikutsertaanProjek:` 1% (3 missing values)
- `JumlahKeterlambatanSebulanTerakhir:` 0.35% (1 missing value)

### Skewness Distribution and Outlier Handling

![image](https://github.com/riyouuyt/Improving-Employee-Retention-by-Predicting-Employee-Attrition-Using-Machine-Learning/assets/122600889/d5d987a5-4293-4cfb-93a8-8ac9e75fad6a)

Given the dataset's limited size, addressing skewness and outliers in numerical features involves using the median to fill in missing data due to their pronounced skewness.

### Duplicate Data
Upon examination, the dataset doesn’t contain any duplicate entries.

### Handling Inconsistent Data
To standardize certain columns:
- The merged '-' values in the marital status column are recategorized as 'other'.
- Values indicating previous work experience (value 1) are unified as 'yes'.
- 'Product design (UI & UX)' within the resignation reason column is consolidated into 'other'.

The result:
![image](https://github.com/riyouuyt/Improving-Employee-Retention-by-Predicting-Employee-Attrition-Using-Machine-Learning/assets/122600889/fc1b544f-869a-4f70-ad63-a20f618df626)


### Feature Engineering
1. **StatusKerja Column:** 
   A new column, `StatusKerja`, is created to indicate current employment status based on the `TanggalResign` column. Employees without a resignation date (`-` or null) are categorized as `Masih Bekerja`, while others are labeled `Tidak Bekerja`.
2. **Datetime Conversion:**
   - `TanggalHiring` and `TanggalResign` columns are transformed into datetime format.
   - Extracted year, month, and day from `TanggalHiring`.
   - Calculated the year of resignation.
3. **Employee Age and Age Category:**
   - The `TanggalLahir` column is converted to datetime, and ages of employees are computed based on the current year.
   - An `UsiaKaryawan` column denotes the age of each employee.
   - Created an `KategoriUsia` column categorizing employees as `Muda` (Young), `Menengah` (Middle-aged), or `Tua` (Old) based on predefined age brackets.
4. **Length of Employment:**
   - `TanggalPenilaianKaryawan` and `TanggalHiring` columns are converted to datetime.
   - A new `LamaMenjabat` column is generated to capture the duration of employment in months by calculating the difference between the evaluation and hiring dates.


## 2. Annual Report on Employee Number Changes 📈

### **Employee Changes Over Time**

![image](https://github.com/riyouuyt/Improving-Employee-Retention-by-Predicting-Employee-Attrition-Using-Machine-Learning/assets/122600889/8ad7a3c7-88ce-40b6-917b-ca9f59d2574c)

Observation:

* **2006-2013**: During these years, the number of incoming 
employees is relatively low, and there is a gradual increase in 
outgoing employees. The total remaining employees are high, 
indicating that the company maintains its workforce.
* **2013-2015**: In 2013, a significant number of employees 
resigned, resulting in a sharp drop in the total remaining 
employees. This trend continues in 2014 and 2015. The 
increase in outgoing employees might be due to various 
factors such as job dissatisfaction, better opportunities 
elsewhere, or organizational changes.
* **2015-2018**: The number of incoming employees remains 
moderate, but the number of outgoing employees remains 
high. The total remaining employees continue to decrease, 
indicating a steady decline in the workforce.
* **2018-2019**: In 2018, the number of incoming employees is 
very low, while the number of outgoing employees is 
significantly higher. This leads to a drastic decrease in the total 
remaining employees. The trend of increasing resignations 
might be attributed to internal issues or external factors 
affecting job stability.


## 3. Resign Reason Analysis 📋

### **percantage of employees that still work**

![image](https://github.com/riyouuyt/Improving-Employee-Retention-by-Predicting-Employee-Attrition-Using-Machine-Learning/assets/122600889/7cae2dae-5980-48d4-80c4-9f37d1bc231c)

Annual Report on employee based on work division:
* Software Engineering are the most dominant division on the company with total of 73,71% of the percentage of the whole division shows how really 
are important Software Engineering roles in our company, with probably higher salaries and as a fundamental job on every IT division is really a good 
reason on why lots of Software Engineering are stil stay on our company.
* Followed by UI & UX with 7,73% , While the rest division are below 6% show some assumptions that this maybe there are some structural hierarchy 
of importance role for the company like the software engineering. 
* Beside the role of importance, higher education and experience can be a solid reason on why that happen like the DevOps, Scrum master, etc. 

### **Incoming and Outgoing Employees Based on Career**

![image](https://github.com/riyouuyt/Improving-Employee-Retention-by-Predicting-Employee-Attrition-Using-Machine-Learning/assets/122600889/65ff0695-9ec3-4678-a8b4-c4ae8e0f3813)

Observation:
* **Fresh graduate programs** have consistently shown the highestretention rates in our company over the past few decades.Several plausible explanations for this trend include the
possibility of new employees having false expectations, seekingjob experience, or encountering job mismatches.

* **Mid-level employees**, with several years of experience, mayconsider leaving for better career growth opportunities in moresenior roles. Factors like limited promotion prospects or lack of
skill development within the company could influence their decision to resign.

* **Senior employees**, who have accumulated significantexperience, may seek retirement, a change in career path, or entrepreneurial ventures. An increased desire for work-life
balance or dissatisfaction with company policies may lead to higher resignation rates among seniorstaff.

### **Outoging Employees based on Performance**

![image](https://github.com/riyouuyt/Improving-Employee-Retention-by-Predicting-Employee-Attrition-Using-Machine-Learning/assets/122600889/65daa7dd-5e06-47bc-8295-1ec36bb8b2c5)

Observation:
*  **Very Good Performance:** Surprisingly, employees with consistently "very good" performance ratings show significantly higher resignation rates. While it's expectedthat top performers are retained, the occurrence of resignations in this group raises
questions. This could be due to employees being headhunted for better opportunities or personal career development goals.

* **Mediocre Performance:** Employees with mediocre performance ratings have relatively high resignation rates, which may be attributed to a lack of motivation or
challenges in their roles. The organization might consider addressing factors affecting
the performance of this group.

* **Good Performance:** Employees with consistently good performance ratings show moderate resignation rates. It's crucial to explore the reasons behind theseresignations, which could involve factors like
limited career growth or a desire for more challenging roles.

* **Very Low Performance:** Employees with persistently low performance ratings also experience high resignation rates. Poor performance is often linked to unsuitability for
roles or a lack of job satisfaction. Addressing these underlying issues is essential to improve retention.

* **Low Performance:** Those with consistently low performance have the highest resignation rates. It is vital for the organization to understand the reasons behind the poor performance and resignations,
* which might include job-role mismatches, lack of motivation, or other workplace challenges

### **Resign Reason Analysis for Employee Attrition Management Strategy**

![image](https://github.com/riyouuyt/Improving-Employee-Retention-by-Predicting-Employee-Attrition-Using-Machine-Learning/assets/122600889/841db11b-ce3f-4cda-bd85-718fd502468b)

Observation:
* **Working Hours (Jam Kerja)**: A significant number of resignations being attributed to working hours may indicate that employees are struggling with work-life balance. Long working hours, excessive overtime, or inconsistent schedules can
lead to burnout and negatively impact an individual's quality of life. When work demands become overwhelming and unsustainable, employees may decide to seek positions with more reasonable working hours.

* **Career Change (Ganti Karir)**: This reason could suggest that employees are either seeking career growth or transitioning to different fields that align better with their long-term aspirations.
It could also reflect that employees are not finding opportunities for advancement or skill development within the company. When such opportunities are limited, employees opt for external career changes.

* **Career Clarity (Kejelasan Karir)**: A lack of career clarity within the organization
might be leading to employee dissatisfaction. When employees are uncertain about their career paths, promotion criteria, or professional development opportunities, it can create frustration and hinder their commitment to the company. As a result,
they may decide to pursue opportunities that provide clearer career trajectories and personal growth prospects

## 4. **Automated Resign Behavior Prediction 🤖**

### **Data Split and Feature Selection Analysis**
For robust model training and evaluation, the dataset of 287 employee records was divided into training (79.79%) and testing (20.21%) subsets. Leveraging various machine learning models, each showcasing distinct strengths, revealed the following insights:

### **Model Comparison**
| ML Method              | Accuracy | Precision | Recall | F1-score | ROC   |
|------------------------|----------|-----------|--------|----------|-------|
| Support Vector Machine | 0.67     | 0.0       | 0.0    | 0.0      | 0.5   |
| Gradient Boosting      | 0.95     | 0.9       | 0.95   | 0.92     | 0.95  |
| Decision Tree          | 0.93     | 0.86      | 0.95   | 0.9      | 0.93  |
| Random Forest          | 0.91     | 1.0       | 0.74   | 0.85     | 0.87  |
| Linear Regression      | 0.67     | 0.5       | 0.11   | 0.17     | 0.53  |

### **Recomendation Model Selection**
The Gradient Boosting model demonstrated exceptional performance, boasting an accuracy of 94.83% and an AUC score of 94.80%. Its predictive capabilities and accuracy in identifying potential resignations make it the preferred model for employee retention prediction.

### **Cross-Validation and Hyperparameter Tuning Analysis**

Through Cross-Validation, the model underwent fine-tuning, showcasing reliable performance metrics:
- Mean Precision: 0.97 (±0.05)
- Mean Recall: 0.96 (±0.07)
- Mean ROC-AUC: 0.97 (±0.03)

### **Hyperparameter Optimization**

![image](https://github.com/riyouuyt/Improving-Employee-Retention-by-Predicting-Employee-Attrition-Using-Machine-Learning/assets/122600889/b09b7b43-ac1e-4ade-8a2b-f9807db450a9)

Optimizing hyperparameters yielded an outstanding AUC score of 1.00! The best hyperparameters for the model included:
- Learning Rate: 0.01
- Max Depth: 4
- Min Samples Leaf: 5
- Min Samples Split: 3
- Number of Estimators: 50
- Subsample: 0.8

These refined parameters ensure the model operates at its peak, enhancing its predictive capabilities for employee retention.

### **Feature Importance**

![image](https://github.com/riyouuyt/Improving-Employee-Retention-by-Predicting-Employee-Attrition-Using-Machine-Learning/assets/122600889/7355b406-49fc-4bb8-b28e-54a52e0ea5fb)

Feature Importance Analysis:

*  **AlasanResign** 🔑: The reason for resigning is a crucial factor that affects employee retention. Understanding the specific reasons
why employees leave can help organizations make targeted improvements to their work environment, which can ultimately reduce employee turnover.

*  **AsalKota Jakarta Selatan** 🏙️: The location from which employees come can be an important factor. In this case, employees from South Jakarta might have different commuting
experiences or preferences that could impact their job satisfaction and, in turn, their likelihood of resigning.

### **ROC Curve Analysis**
![image](https://github.com/riyouuyt/Improving-Employee-Retention-by-Predicting-Employee-Attrition-Using-Machine-Learning/assets/122600889/3ec6fda9-e2ff-4952-947c-c261588c0149)

The ROC curve for the model achieved an outstanding area under the curve (AUC) of 0.99, which means the model has a high ability to distinguish between employees who are likely to resign and those
who are likely to stay. This high AUC score is a promising sign that the model is effective in identifying potential resignations, which can be invaluable for employee retention strategies.🚀


## **Presenting Machine Learning Products to the Business Users**

### **SHAP Values using Gradient Boosting**

![image](https://github.com/riyouuyt/Improving-Employee-Retention-by-Predicting-Employee-Attrition-Using-Machine-Learning/assets/122600889/274aa0ec-de3f-441e-b8cf-8937b095382a) 

### **SHAP Values using Neural Network**

![image](https://github.com/riyouuyt/Improving-Employee-Retention-by-Predicting-Employee-Attrition-Using-Machine-Learning/assets/122600889/099aa595-bfc8-444c-8781-7846b3f5e147)

### **SHAP Values Analysis** 

Both the Gradient Boosting and Neural Network models agree on the importance of factors such as `AlasanResign` and `UsiaKaryawan` as determinants of employee retention. Resignation reasons should be closely 
monitored, and employee tenure needs to be nurtured and rewarded. However, the neural network emphasizes the significance of `LamaMenjabat` and `HariHiring`, indicating that nurturing employee growth and 
optimizing onboarding processes are essential for retention. 

Incorporating insights from both models can lead to a more comprehensive approach to employee retention. By addressing these key features, 
companies can proactively reduce turnover, boost employee satisfaction, and ultimately enhance their business performance.





