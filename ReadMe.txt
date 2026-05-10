//// PYTHON FILES OVERVIEW

1. Generate_dataset.py generates a new dataset from AI4I2020 predictive maintenance dataset.
2. Simulate_and_monitor.py simulates sensor behavior by inserting data into the database 1 row per second and sends alearts if machine health is critical
3. train_model.py is used to train model on generated dataset.
4. Multi_machine_data.xlss is the training dataset.
4. Simulation_data.xlss is the test data.

//// DATABASE FILES

1. SQL5 Query 5 holes the database structure queries.

//// MAIN DASHBOARD FILES

1. Machine Doctor fyp Dashboard is the main file connected to the Database and Montiors the machines health.


// SETUP INSTRUCTIONS

1. run sql5 file and run the whole file in SSMS. (creates full database)
2. run generate_dataset file to generate a dataset.
3. run train_model to train the model and it will produce (.keras,.pkl) & simulation_data.xlss files.
4. run Simulate_and_monitor.py , it inserts one row per second in the database to simulate a real sensor.
5. open .pbix (dashboard) file in Microsoft PowerBI, go to connect sourse options, select SQL Server, put in login credentials and connect to server, click direct query.
6. ALL SET 

// FOR REMOT ACESS

You can publish your report to power bi service for remort acess.

// TECNOLOGY STACK

1. Power BI
2. Python version 10
3. SQL server