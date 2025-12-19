import pickle
import pandas as pd

def main()
    #Loading model
    with open("30-diamond_model_completed_pkl","rb") as f:
        saved_data=pickle.load(f)
    model = saved_data["model"]
    X_test=pd.read_csv("30-testdata.csv")
    print(model.predict(X_test))

if __name__=="__main__":
    main()
