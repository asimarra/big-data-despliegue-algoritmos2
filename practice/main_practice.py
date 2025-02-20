import methods_for_practice as func

def main():
    args_values = func.ask_params()
    df = func.load_clean_dataset()
    x_train, x_test, y_train, y_test = func.data_treatment(df)
    func.mlflow_tracking_with_pipeline(args_values.job_name, x_train, x_test, y_train, y_test, args_values)

if __name__ == "__main__":
    main()

