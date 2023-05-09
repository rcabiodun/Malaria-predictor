import random


def clean_csv_file(df):
    df.pop("Unnamed: 133")

    for index in df.loc[df["prognosis"]=="Malaria"].index:
        df["loss_of_appetite"][index]=1
        df["nausea"][index]=random.randint(0,1)
        df["vomiting"][index]=random.randint(0,1)
        df["sweating"][index]=random.randint(0,1)
        df["fatigue"][index]=1
        df["skin_rash"][index]=0
        df["itching"][index]=0
        df["high_fever"][index]=1

    for index in df.loc[df["prognosis"]=="AIDS"].index:
        df["loss_of_appetite"][index]=1
        df["nausea"][index]=1
        df["vomiting"][index]=1
        df["sweating"][index]=random.randint(0,1)
        df["fatigue"][index]=1
        df["cough"][index]=1
        df["skin_rash"][index]=1
        df["high_fever"][index]=1
        df["muscle_wasting"][index]=random.randint(0,1)
        df["swelled_lymph_nodes"]=1
        df["diarrhoea"]=1
        df["stomach_pain"]=1
        df["ulcers_on_tongue"]=1

    for index in df.loc[df["prognosis"]=="Chicken pox"].index:
        df["loss_of_appetite"][index]=random.randint(0,1)
        df["blister"][index]=1
        df["high_fever"][index]=1
        df["fatigue"][index]=1
        df["headache"][index]=1
        df["itching"][index]=1
        df["sweating"][index]=0
        df["swelled_lymph_nodes"]=random.randint(0,1)
        df["skin_rash"][index]=1
        df["red_spots_over_body"][index]=random.randint(0,1)
    
    for index in df.loc[df["prognosis"]=="Tuberculosis"].index:
        df["loss_of_appetite"][index]=1
        df["sweating"][index]=random.randint(0,1)
        df["cough"][index]=1
        df["chest_pain"][index]=1
        df["muscle_wasting"][index]=1
        df["high_fever"][index]=1
        df["chills"][index]=1
        df["fatigue"][index]=1
        df["swelled_lymph_nodes"]=random.randint(0,1)


    

