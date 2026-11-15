import streamlit as st  # create streamlit interface
import pandas as pd  # dataframes and all
import joblib  # load the .pkl file

# load pipeline
pipeline = joblib.load('models/employee_attrition_pipeline.pkl')

# configure the web browser view
st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="📊",
    layout="wide"
)

# main heading
st.title("Employee Attrition Prediction")

# intro
st.write(
    "Predict whether an employee is likely to leave the organization "
    "based on their professional and personal attributes."
)

# now the user input should conatin the same features as in X_raw

# we will use form here to accept employee data as if we dont use it the app will rerun everytime the user changes the inputs
# when we use form the app ruuns when the predict button is pressed


with st.form("employee_form"):  # name of form

    # numerical cols
    st.subheader("Baisc Employee Details") # heading

    col1, col2, col3 = st.columns(3)  # divide the interface layout in 3 cols

    with col1:  # in 1st col accept age details
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=60,
            value=30
        )
 
    with col2:  # 2nd col accept daily rate details
        daily_rate = st.number_input(
            "Daily Rate",
            min_value=0,
            value=800
        )

    with col3:  # 3rd col accept distance details
        distance = st.number_input(
            "Distance From Home",
            min_value=1,
            value=5
        )
        
    # categorical cols
    # 6 nominal and 1 ordinal
    st.subheader("Categorical Details")

    col1, col2, col3 = st.columns(3)

    with col1:  # department and edu field
        department = st.selectbox(
            "Department",
            ["Sales", "Research & Development", "Human Resources"]
        )

        education_field = st.selectbox(
            "Education Field",
            [
                "Life Sciences",
                "Medical",
                "Marketing",
                "Technical Degree",
                "Other",
                "Human Resources"
            ]
        )

    with col2:  # gender and job role
        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        job_role = st.selectbox(
            "Job Role",
            [
                "Healthcare Representative",
                "Human Resources",
                "Laboratory Technician",
                "Manager",
                "Manufacturing Director",
                "Research Director",
                "Research Scientist",
                "Sales Executive",
                "Sales Representative"
            ]
        )

    with col3:  # marital status,overtime,business travel(ordinal)
        marital_status = st.selectbox(
            "Marital Status",
            ["Single", "Married", "Divorced"]
        )

        overtime = st.selectbox(
            "OverTime",
            ["Yes", "No"]
        )

        business_travel = st.selectbox(
            "Business Travel",
            ["Non-Travel", "Travel_Rarely", "Travel_Frequently"]
        )
        
        
    # other numerical input cols 
    st.subheader("Additional Employee Metrics")

    col1, col2, col3 = st.columns(3)

    with col1:
        education = st.number_input("Education", 1, 5, 3)  # values from 1 to 5 default set 3
        environment_satisfaction = st.number_input("Environment Satisfaction", 1, 4, 3) # values from 1 to 4 default set 3
        hourly_rate = st.number_input("Hourly Rate", 30, 100, 65)
        job_involvement = st.number_input("Job Involvement", 1, 4, 3)
        job_level = st.number_input("Job Level", 1, 5, 2)
        job_satisfaction = st.number_input("Job Satisfaction", 1, 4, 3)
        monthly_income = st.number_input("Monthly Income", 1009, 19999, 5000)
        monthly_rate = st.number_input("Monthly Rate", 2094, 26999, 14000)

    with col2:
        num_companies_worked = st.number_input("Number of Companies Worked", 0, 9, 2)
        percent_salary_hike = st.number_input("Percent Salary Hike", 11, 25, 15)
        performance_rating = st.number_input("Performance Rating", 3, 4, 3)
        relationship_satisfaction = st.number_input("Relationship Satisfaction", 1, 4, 3)
        stock_option_level = st.number_input("Stock Option Level", 0, 3, 1)
        total_working_years = st.number_input("Total Working Years", 0, 40, 10)

    with col3:
        training_times_last_year = st.number_input("Training Times Last Year", 0, 6, 3)
        work_life_balance = st.number_input("Work Life Balance", 1, 4, 3)
        years_at_company = st.number_input("Years At Company", 0, 40, 5)
        years_in_current_role = st.number_input("Years In Current Role", 0, 18, 3)
        years_since_last_promotion = st.number_input("Years Since Last Promotion", 0, 15, 1)
        years_with_curr_manager = st.number_input("Years With Current Manager", 0, 17, 3)

    submitted = st.form_submit_button("Predict Attrition")
   
if submitted:

    # create dataframe out of given inputs and give same col names as in X_raw as coltransformer identifies what preprocessing
    # to do based on the col names it learned when model was trained by fitting on X_raw_train and y_train
    input_data = pd.DataFrame([{
        # same col names as X_raw and ip we took with above form
        'Age': age,  
        'BusinessTravel': business_travel,
        'DailyRate': daily_rate,
        'Department': department,
        'DistanceFromHome': distance,
        'Education': education,
        'EducationField': education_field,
        'EnvironmentSatisfaction': environment_satisfaction,
        'Gender': gender,
        'HourlyRate': hourly_rate,
        'JobInvolvement': job_involvement,
        'JobLevel': job_level,
        'JobRole': job_role,
        'JobSatisfaction': job_satisfaction,
        'MaritalStatus': marital_status,
        'MonthlyIncome': monthly_income,
        'MonthlyRate': monthly_rate,
        'NumCompaniesWorked': num_companies_worked,
        'OverTime': overtime,
        'PercentSalaryHike': percent_salary_hike,
        'PerformanceRating': performance_rating,
        'RelationshipSatisfaction': relationship_satisfaction,
        'StockOptionLevel': stock_option_level,
        'TotalWorkingYears': total_working_years,
        'TrainingTimesLastYear': training_times_last_year,
        'WorkLifeBalance': work_life_balance,
        'YearsAtCompany': years_at_company,
        'YearsInCurrentRole': years_in_current_role,
        'YearsSinceLastPromotion': years_since_last_promotion,
        'YearsWithCurrManager': years_with_curr_manager
    }])
    
    # validate following things to avoid inconsistent inputs
    
    # YearsAtCompany cannot exceed TotalWorkingYears.
    # YearsInCurrentRole cannot exceed YearsAtCompany.
    # YearsSinceLastPromotion cannot exceed YearsAtCompany.
    # YearsWithCurrManager cannot exceed YearsAtCompany.
    
    errors = []

    if years_at_company > total_working_years:
        errors.append("Years at company cannot exceed total working years.")

    if years_in_current_role > years_at_company:
        errors.append("Years in current role cannot exceed years at company.")

    if years_since_last_promotion > years_at_company:
        errors.append("Years since last promotion cannot exceed years at company.")

    if years_with_curr_manager > years_at_company:
        errors.append("Years with current manager cannot exceed years at company.")

    if errors:
        for error in errors:
            st.error(error)
            
            
    else:
        
        # show the user entered details
        st.write("Employee Input:")
        st.dataframe(input_data)
        
        # generate prediction by passing the dataframe input_data to pipeline which we have trained
    
        prediction = pipeline.predict(input_data)

        if prediction[0] == 1:
            st.error("Prediction: Employee is likely to leave.")
        else:
            st.success("Prediction: Employee is likely to stay.")
        
        
        #    now probability of prediction i.e prob of each class(0 and 1)
        probabilities = pipeline.predict_proba(input_data)[0]
        # we use [0] because pipeline.predict_proba returns a 2d array [[]] where the outerlist contains various lists(each representing
        # a row of the test data and in each of the inner list there is prob of classes i.e class 0 and class 1)
        # eg probabilities here looks like this say[[0.25,0.75]]
        # if there were two input rows like if the df input_data had 2 ip rows it would look like[[0.25,0.75],[0.3,0.7]]
        # so here [0] extracts first row details or prediction probabilities of first employee details passed in input_data
    
        stay_probability = probabilities[0] * 100  # prob of class 0
        leave_probability = probabilities[1] * 100  # prob of class 1
    
        st.subheader("Prediction Probabilities")
        
        st.caption(
        "These probabilities are estimates generated by the trained machine learning model. "
        "They indicate the model's confidence in each prediction and are not guarantees "
        "of an employee's future behavior."
        )
    
        col1, col2 = st.columns(2)
    
        with col1:
            st.metric(
                "Probability of Staying",
                f"{stay_probability:.2f}%"
            )
    
        with col2:
            st.metric(
                "Probability of Leaving",
                f"{leave_probability:.2f}%"
            )
        
# Flow

# user enters the details through form
# this data is passed to create pandas df
# this df is passed to trained pipeline
# the preprocessor i.e coltransformer transforms cols on basis of learned categories (for ordinal ordinal encoding and for nominal OHE)
# xgboost then predicts the results either 0(stay) or 1(leave)
# predictions store in prediction then display result